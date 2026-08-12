#!/usr/bin/env python3
"""
st06_scale.py — определение масштабного коэффициента модели по линейному эталону.

Детектирует маркеры ArUco на кадрах, выполняет пространственную засечку их
центров по параметрам камер, восстановленным COLMAP, и вычисляет масштабный
коэффициент по фактически измеренным расстояниям между центрами маркеров.

Модель COLMAP должна быть в текстовом формате:
    colmap model_converter --input_path sparse/0 --output_path sparse/0_txt \\
        --output_type TXT

Оценка погрешности:
  - при трёх и более маркерах — по разбросу частных значений k, вычисленных
    для разных пар;
  - при двух маркерах (единственная база) — по разбросу значений k, полученных
    при повторной засечке по случайным подмножествам кадров.

Зависимости: numpy, opencv-contrib-python
"""

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Чтение модели COLMAP (текстовый формат)
# ---------------------------------------------------------------------------

def read_cameras(path):
    """Читает cameras.txt -> {camera_id: (model, w, h, params)}."""
    cameras = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        cameras[cam_id] = {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": np.array([float(p) for p in parts[4:]]),
        }
    return cameras


def read_images(path):
    """Читает images.txt -> {image_name: {camera_id, R, t}}.

    COLMAP хранит поворот и перенос в направлении мир -> камера:
    X_cam = R @ X_world + t

    Каждому изображению соответствуют две строки: параметры и список
    двумерных точек, который может быть пустым. Строки параметров
    распознаются по структуре, а не по чётности номера.
    """
    images = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 10:
            continue  # строка списка двумерных точек
        if Path(parts[9]).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            qw, qx, qy, qz = (float(x) for x in parts[1:5])
            tx, ty, tz = (float(x) for x in parts[5:8])
            cam_id = int(parts[8])
        except ValueError:
            continue
        images[parts[9]] = {
            "camera_id": cam_id,
            "R": quat_to_rot(qw, qx, qy, qz),
            "t": np.array([tx, ty, tz]),
        }
    return images


def quat_to_rot(qw, qx, qy, qz):
    """Кватернион (w, x, y, z) -> матрица поворота 3x3."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def camera_matrices(cam):
    """Возвращает (K, dist) по описанию камеры COLMAP."""
    p, model = cam["params"], cam["model"]

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        fx = fy = f
        dist = np.zeros(5)
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        dist = np.zeros(5)
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p[:4]
        fx = fy = f
        dist = np.array([k1, 0, 0, 0, 0], dtype=float)
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = p[:5]
        fx = fy = f
        dist = np.array([k1, k2, 0, 0, 0], dtype=float)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        dist = np.array([k1, k2, p1, p2, 0], dtype=float)
    else:
        raise ValueError(
            f"Модель камеры {model} не поддерживается. "
            f"Используйте SIMPLE_RADIAL, RADIAL, PINHOLE или OPENCV."
        )

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    return K, dist


# ---------------------------------------------------------------------------
# Детектирование маркеров
# ---------------------------------------------------------------------------

def make_detector(dict_name):
    """Создаёт детектор ArUco с учётом версии OpenCV."""
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dict_name)
    )
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda img: detector.detectMarkers(img)[:2]

    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return lambda img: cv2.aruco.detectMarkers(
        img, dictionary, parameters=params
    )[:2]


def quad_center(corners):
    """Точка пересечения диагоналей четырёхугольника.

    Для плоского маркера это проекция его геометрического центра —
    в отличие от среднего углов, которое при перспективе смещено.
    """
    p0, p1, p2, p3 = [np.asarray(c, dtype=float) for c in corners]
    d1, d2 = p2 - p0, p3 - p1
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = p1 - p0
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return (p0 + p1 + p2 + p3) / 4.0
    s = np.linalg.solve(A, b)
    return p0 + s[0] * d1


def detect_markers(frames_dir, detector, images, cameras):
    """Собирает наблюдения центров маркеров по всем кадрам.

    -> {marker_id: [(image_name, (u, v)), ...]}
    """
    observations = {}
    frames = sorted(
        p for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    scanned = 0

    for path in frames:
        if path.name not in images:
            continue  # кадр не вошёл в реконструкцию
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        scanned += 1

        corners, ids = detector(img)
        if ids is None:
            continue
        for quad, mid in zip(corners, ids.flatten()):
            uv = quad_center(quad.reshape(4, 2))
            observations.setdefault(int(mid), []).append((path.name, uv))

    return observations, scanned, len(frames)


# ---------------------------------------------------------------------------
# Пространственная засечка
# ---------------------------------------------------------------------------

def triangulate(obs, images, cameras, max_reproj_px=None, min_views=4):
    """Засечка точки по нескольким кадрам методом наименьших квадратов (DLT).

    При заданном max_reproj_px выполняется итеративная отбраковка наблюдений
    с невязкой репроекции выше порога: среднеквадратическая невязка
    чувствительна к выбросам, и несколько кадров со смазом способны исказить
    результат при в целом качественных исходных данных.

    Возвращает словарь с координатами, статистикой невязок и числом
    отбракованных наблюдений либо None.
    """
    used = list(obs)
    rejected = 0

    for _ in range(5):
        result = _solve(used, images, cameras)
        if result is None:
            return None
        X, residuals = result

        if max_reproj_px is None or len(used) <= min_views:
            break

        keep = [o for o, r in zip(used, residuals) if r <= max_reproj_px]
        if len(keep) == len(used) or len(keep) < min_views:
            break
        rejected += len(used) - len(keep)
        used = keep

    residuals = np.asarray(residuals)
    return {
        "X": X,
        "n_views": len(used),
        "n_rejected": rejected,
        "rms_px": float(np.sqrt(np.mean(residuals ** 2))),
        "median_px": float(np.median(residuals)),
        "p90_px": float(np.percentile(residuals, 90)),
        "max_px": float(residuals.max()),
    }


def _solve(obs, images, cameras):
    """Одна итерация засечки. Возвращает (X, невязки по наблюдениям)."""
    rows, proj = [], []

    for name, uv in obs:
        img = images[name]
        cam = cameras[img["camera_id"]]
        K, dist = camera_matrices(cam)

        # Приведение к нормализованным координатам с учётом дисторсии
        pt = np.array([[uv]], dtype=np.float64)
        norm = cv2.undistortPoints(pt, K, dist).reshape(2)

        P = np.hstack([img["R"], img["t"].reshape(3, 1)])
        proj.append((P, norm, K))

        x, y = norm
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])

    if len(proj) < 2:
        return None

    _, _, Vt = np.linalg.svd(np.array(rows))
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    X = Xh[:3] / Xh[3]

    residuals = []
    for P, norm, K in proj:
        cam_pt = P @ np.append(X, 1.0)
        if cam_pt[2] <= 0:
            residuals.append(np.inf)
            continue
        f = np.array([K[0, 0], K[1, 1]])
        residuals.append(
            float(np.linalg.norm((cam_pt[:2] / cam_pt[2] - norm) * f))
        )

    return X, residuals


def check_image_sizes(frames_dir, images, cameras, limit=3):
    """Сверяет размеры кадров на диске с параметрами камер модели.

    Расхождение означает, что детектирование маркеров выполняется на
    изображениях иного масштаба, чем тот, которому соответствуют
    внутренние параметры камер, что вносит систематическую погрешность.
    """
    problems = []
    for name in list(images)[:limit]:
        path = Path(frames_dir) / name
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cam = cameras[images[name]["camera_id"]]
        h, w = img.shape[:2]
        if (w, h) != (cam["width"], cam["height"]):
            problems.append((name, (w, h), (cam["width"], cam["height"])))
    return problems


# ---------------------------------------------------------------------------
# Масштабный коэффициент
# ---------------------------------------------------------------------------

def scale_from_points(points, measured_mm):
    """Частные значения k по всем парам маркеров с известным расстоянием."""
    result = []
    for a, b in combinations(sorted(points), 2):
        key = f"{a}-{b}"
        alt = f"{b}-{a}"
        d_ref = measured_mm.get(key, measured_mm.get(alt))
        if d_ref is None:
            continue
        d_model = float(np.linalg.norm(points[a] - points[b]))
        if d_model <= 0:
            continue
        result.append({
            "pair": key,
            "measured_mm": d_ref,
            "model_units": d_model,
            "k_mm_per_unit": d_ref / d_model,
        })
    return result


def bootstrap_scale(observations, images, cameras, measured_mm,
                    max_reproj=None, n_iter=50, frac=0.7, seed=0):
    """Оценка устойчивости k к составу кадров (для схемы с двумя маркерами)."""
    rng = random.Random(seed)
    values = []

    for _ in range(n_iter):
        subset_points = {}
        for mid, obs in observations.items():
            n_keep = max(2, int(round(len(obs) * frac)))
            sample = rng.sample(obs, min(n_keep, len(obs)))
            tri = triangulate(sample, images, cameras, max_reproj)
            if tri is not None:
                subset_points[mid] = tri["X"]
        pairs = scale_from_points(subset_points, measured_mm)
        if pairs:
            values.append(np.mean([p["k_mm_per_unit"] for p in pairs]))

    if len(values) < 2:
        return None
    return {
        "n_iterations": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "rel_std_pct": float(100 * np.std(values, ddof=1) / np.mean(values)),
    }


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("frames", help="каталог с кадрами")
    ap.add_argument("model", help="каталог модели COLMAP в текстовом формате")
    ap.add_argument("--project", help="project.json — источник measured_distances_mm")
    ap.add_argument("--distances", help="расстояния между центрами, мм, "
                                       "например '0-1=998.5,0-2=1997.0'")
    ap.add_argument("--dict", default="DICT_5X5_50", help="словарь ArUco")
    ap.add_argument("--max-reproj", type=float, default=3.0,
                    help="порог отбраковки наблюдений по невязке репроекции, "
                         "пикс. (0 — не отбраковывать)")
    ap.add_argument("--bootstrap", type=int, default=50,
                    help="число итераций оценки устойчивости (0 — отключить)")
    ap.add_argument("--report", help="путь для сохранения отчёта в JSON")
    args = ap.parse_args()

    model_dir = Path(args.model)
    cameras = read_cameras(model_dir / "cameras.txt")
    images = read_images(model_dir / "images.txt")

    # Фактические расстояния между центрами маркеров
    measured = {}
    if args.project:
        data = json.loads(Path(args.project).read_text(encoding="utf-8"))
        measured = (data.get("etalon", {}).get("measured_distances_mm") or {})
    if args.distances:
        for item in args.distances.split(","):
            key, value = item.split("=")
            measured[key.strip()] = float(value)
    if not measured:
        sys.exit(
            "Не заданы фактические расстояния между центрами маркеров. "
            "Заполните measured_distances_mm в project.json либо укажите "
            "--distances."
        )

    if args.max_reproj <= 0:
        args.max_reproj = None

    detector = make_detector(args.dict)
    observations, scanned, total = detect_markers(
        args.frames, detector, images, cameras
    )

    print(f"Кадров в каталоге:        {total}")
    print(f"Кадров в реконструкции:   {scanned}")
    if not observations:
        sys.exit("Маркеры не обнаружены ни на одном кадре.")

    problems = check_image_sizes(args.frames, images, cameras)
    if problems:
        print("\nВНИМАНИЕ: размеры кадров не совпадают с параметрами камер "
              "модели:", file=sys.stderr)
        for name, actual, model in problems:
            print(f"  {name}: файл {actual[0]}x{actual[1]}, "
                  f"модель {model[0]}x{model[1]}", file=sys.stderr)
        print("  Детектирование маркеров и внутренние параметры камер "
              "относятся к разным масштабам изображения.\n", file=sys.stderr)

    print(f"\nОбнаружено маркеров:      {len(observations)}")
    points, per_marker = {}, {}
    for mid in sorted(observations):
        obs = observations[mid]
        tri = triangulate(obs, images, cameras, args.max_reproj)
        if tri is None:
            print(f"  ID {mid}: наблюдений {len(obs)} — засечка не выполнена")
            continue
        points[mid] = tri["X"]
        per_marker[mid] = {k: v for k, v in tri.items() if k != "X"}
        extra = (f", отбраковано {tri['n_rejected']}"
                 if tri["n_rejected"] else "")
        print(f"  ID {mid}: кадров {tri['n_views']:>3}{extra}; невязка "
              f"медиана {tri['median_px']:.2f} / p90 {tri['p90_px']:.2f} / "
              f"макс {tri['max_px']:.2f} px")

    if len(points) < 2:
        sys.exit("Для определения масштаба требуется не менее двух маркеров.")

    pairs = scale_from_points(points, measured)
    if not pairs:
        sys.exit(
            "Ни для одной пары обнаруженных маркеров не заданы фактические "
            "расстояния. Проверьте ключи в measured_distances_mm."
        )

    k_values = [p["k_mm_per_unit"] for p in pairs]
    k_mean = float(np.mean(k_values))

    print(f"\nЧастные значения масштабного коэффициента:")
    for p in pairs:
        print(f"  {p['pair']}: {p['measured_mm']:>8.1f} мм / "
              f"{p['model_units']:.6f} ед. = {p['k_mm_per_unit']:.3f} мм/ед.")

    report = {
        "stage": "st06_scale",
        "frames_total": total,
        "frames_in_model": scanned,
        "markers": per_marker,
        "pairs": pairs,
        "scale_factor_mm_per_unit": k_mean,
    }

    if len(k_values) >= 2:
        std = float(np.std(k_values, ddof=1))
        rel = 100 * std / k_mean
        report["scale_rmse_pct"] = rel
        report["rmse_source"] = "pairs"
        print(f"\nМасштабный коэффициент:   {k_mean:.4f} мм/ед.")
        print(f"Разброс по парам:         {std:.4f} мм/ед. ({rel:.3f} %)")
    else:
        print(f"\nМасштабный коэффициент:   {k_mean:.4f} мм/ед.")
        print("Единственная база — разброс по парам не определён.")
        if args.bootstrap > 0:
            bs = bootstrap_scale(
                observations, images, cameras, measured,
                max_reproj=args.max_reproj, n_iter=args.bootstrap
            )
            if bs:
                report["scale_rmse_pct"] = bs["rel_std_pct"]
                report["rmse_source"] = "bootstrap"
                report["bootstrap"] = bs
                print(f"Оценка по подмножествам кадров "
                      f"({bs['n_iterations']} итераций):")
                print(f"  среднее {bs['mean']:.4f} мм/ед., "
                      f"СКО {bs['std']:.4f} мм/ед. ({bs['rel_std_pct']:.3f} %)")

    # Контроль систематической зависимости от длины базы
    if len(pairs) >= 3:
        ordered = sorted(pairs, key=lambda p: p["measured_mm"])
        ks = [p["k_mm_per_unit"] for p in ordered]
        monotone = all(a <= b for a, b in zip(ks, ks[1:])) or \
                   all(a >= b for a, b in zip(ks, ks[1:]))
        report["monotone_with_base_length"] = bool(monotone)
        if monotone:
            print("\nВНИМАНИЕ: частные значения k монотонно изменяются "
                  "с ростом длины базы — признак неучтённой дисторсии.",
                  file=sys.stderr)

    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nОтчёт сохранён: {args.report}")


if __name__ == "__main__":
    main()