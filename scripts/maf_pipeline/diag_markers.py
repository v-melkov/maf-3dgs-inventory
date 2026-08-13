#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_markers.py — независимая диагностика наблюдений ArUco по модели COLMAP.

Скрипт НЕ зависит от st08_scale.py: он сам детектирует маркеры в кадрах,
сам триангулирует углы маркеров по позам камер из sparse-модели и сам
считает невязки репроекции. Нужен для двух вещей:

  1. Разделить гипотезы о причине больших невязок:
     - невязка растёт с радиусом от главной точки  -> модель дисторсии
       (SIMPLE_RADIAL) недостаточна, помогает --camera-model OPENCV;
     - невязка кучкуется по номерам кадров / коррелирует со смещением
       камеры между кадрами -> электронная стабилизация или смаз,
       моделью камеры не лечится.

  2. Получить масштабные коэффициенты по базам разной длины:
     - стороны маркера (номинал --marker-size, по умолчанию 100 мм),
     - диагонали маркера (marker-size * sqrt(2)),
     - расстояния между центрами маркеров (--base ID1-ID2=мм).

Выход:
  reports/diag_markers.csv   — по одному наблюдению на строку
  reports/diag_markers.json  — агрегаты, оценки масштаба по базам
  reports/diag_markers.png   — 4 графика (если есть matplotlib)

Пример:
  python diag_markers.py --project projects/bench --base 1-2=1000
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- утилиты

def die(msg):
    print("ОШИБКА: " + msg, file=sys.stderr)
    sys.exit(1)


def frame_index(name):
    """Порядковый номер кадра из имени файла (последняя группа цифр)."""
    m = re.findall(r"(\d+)", str(name))
    return int(m[-1]) if m else -1


def attr_or_call(obj, names):
    """Совместимость с разными версиями pycolmap: атрибут либо метод."""
    for n in names:
        v = getattr(obj, n, None)
        if v is None:
            continue
        if callable(v):
            try:
                return v()
            except TypeError:
                continue
        return v
    return None


def rigid_to_Rt(rigid):
    """Rigid3d -> (R 3x3, t 3)."""
    M = attr_or_call(rigid, ["matrix"])
    if M is not None:
        M = np.asarray(M, dtype=float)
        if M.shape == (3, 4):
            return M[:, :3], M[:, 3]
        if M.shape == (4, 4):
            return M[:3, :3], M[:3, 3]
    rot = getattr(rigid, "rotation", None)
    t = np.asarray(getattr(rigid, "translation"), dtype=float).reshape(3)
    R = attr_or_call(rot, ["matrix"])
    if R is None:
        die("не удалось получить матрицу поворота из позы камеры")
    return np.asarray(R, dtype=float).reshape(3, 3), t


# ---------------------------------------------------------------- COLMAP

def load_reconstruction(sparse_dir):
    try:
        import pycolmap
    except ImportError:
        die("не установлен pycolmap")
    rec = pycolmap.Reconstruction(str(sparse_dir))
    images = {}
    for image_id, img in rec.images.items():
        has_pose = attr_or_call(img, ["has_pose", "registered"])
        if has_pose is False:
            continue
        rigid = attr_or_call(img, ["cam_from_world"])
        if rigid is None:
            continue
        R, t = rigid_to_Rt(rigid)
        images[str(img.name)] = {
            "image_id": image_id,
            "camera_id": img.camera_id,
            "R": R,
            "t": t,
            "center": (-R.T @ t),
        }
    if not images:
        die("в модели нет зарегистрированных изображений")
    return rec, images


def _as_2d(out, n):
    """Результат pycolmap -> массив n x 2, None -> NaN."""
    if out is None:
        return np.full((n, 2), np.nan)
    arr = np.asarray([[np.nan, np.nan] if r is None else np.asarray(r, dtype=float).ravel()[:2]
                      for r in (out if np.ndim(out) > 1 or isinstance(out, (list, tuple))
                                else [out])], dtype=float)
    return arr.reshape(-1, 2)


def make_camera_ops(camera):
    """Возвращает (to_normalized, to_pixel, principal_point).

    to_normalized: пиксели (N x 2) -> нормированные координаты (N x 2)
    to_pixel:      точки в системе камеры (N x 3) -> пиксели (N x 2)

    API pycolmap менялось: img_from_cam в одних версиях принимает
    нормированные 2D, в других — 3D в системе камеры. Пробуем оба.
    """
    f_cam_from_img = getattr(camera, "cam_from_img", None)
    f_img_from_cam = getattr(camera, "img_from_cam", None)
    if not callable(f_cam_from_img) or not callable(f_img_from_cam):
        die("в pycolmap.Camera нет методов cam_from_img / img_from_cam")

    def to_normalized(uv):
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        try:
            return _as_2d(f_cam_from_img(uv), len(uv))
        except TypeError:
            return _as_2d([f_cam_from_img(p) for p in uv], len(uv))

    def to_pixel(xyz):
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        for call in (
            lambda: f_img_from_cam(xyz),                          # 3D пакетом
            lambda: [f_img_from_cam(p) for p in xyz],             # 3D по одной
            lambda: f_img_from_cam(xyz[:, :2] / xyz[:, 2:3]),     # нормированные 2D
        ):
            try:
                return _as_2d(call(), len(xyz))
            except TypeError:
                continue
        die("не подобран вызов img_from_cam для этой версии pycolmap")

    pp = attr_or_call(camera, ["principal_point"])
    if pp is None:
        pp = np.array([camera.width / 2.0, camera.height / 2.0])
    pp = np.asarray(pp, dtype=float).reshape(2)
    return to_normalized, to_pixel, pp


# ---------------------------------------------------------------- ArUco

def build_detector(dict_name):
    try:
        import cv2
    except ImportError:
        die("не установлен opencv-python (нужен opencv-contrib-python для aruco)")
    if not hasattr(cv2, "aruco"):
        die("в сборке OpenCV нет модуля aruco — поставьте opencv-contrib-python")
    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        die("неизвестный словарь ArUco: " + dict_name)
    dict_id = getattr(aruco, dict_name)
    if hasattr(aruco, "getPredefinedDictionary"):
        adict = aruco.getPredefinedDictionary(dict_id)
    else:
        adict = aruco.Dictionary_get(dict_id)

    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        # субпиксельное уточнение углов — критично для метрики
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        det = aruco.ArucoDetector(adict, params)

        def detect(gray):
            return det.detectMarkers(gray)
    else:
        params = aruco.DetectorParameters_create()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        def detect(gray):
            return aruco.detectMarkers(gray, adict, parameters=params)

    return cv2, detect


def detect_all(frames_dir, images, dict_name, exts):
    cv2, detect = build_detector(dict_name)
    obs = defaultdict(list)  # (marker_id, corner_idx) -> [(image_name, u, v)]
    n_frames = 0
    for name in sorted(images.keys()):
        path = Path(frames_dir) / name
        if not path.exists():
            for e in exts:
                cand = Path(frames_dir) / (Path(name).stem + e)
                if cand.exists():
                    path = cand
                    break
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        n_frames += 1
        corners, ids, _ = detect(img)
        if ids is None:
            continue
        for quad, mid in zip(corners, ids.flatten()):
            pts = np.asarray(quad, dtype=float).reshape(4, 2)
            for k in range(4):
                obs[(int(mid), k)].append((name, pts[k, 0], pts[k, 1]))
    return obs, n_frames


# ---------------------------------------------- триангуляция и невязки

def triangulate(rays, poses):
    """DLT по нормализованным координатам. rays: Nx2, poses: [(R,t)]."""
    A = []
    for (x, y), (R, t) in zip(rays, poses):
        P = np.hstack([R, t.reshape(3, 1)])
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def triangulate_irls(rays, poses, uv, to_pixel, huber=1.5, iters=6):
    X = triangulate(rays, poses)
    if X is None:
        return None, None
    w = np.ones(len(rays))
    for _ in range(iters):
        A, rows_w = [], []
        for (x, y), (R, t), wi in zip(rays, poses, w):
            P = np.hstack([R, t.reshape(3, 1)])
            A.append(wi * (x * P[2] - P[0]))
            A.append(wi * (y * P[2] - P[1]))
            rows_w.extend([wi, wi])
        A = np.asarray(A, dtype=float)
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        if abs(Xh[3]) < 1e-12:
            break
        X = Xh[:3] / Xh[3]
        res = residuals(X, poses, uv, to_pixel)
        res_w = np.where(np.isfinite(res), res, huber)
        w = np.where(res_w <= huber, 1.0, huber / np.maximum(res_w, 1e-9))
    return X, residuals(X, poses, uv, to_pixel)


def residuals(X, poses, uv, to_pixel):
    out = np.full(len(poses), np.nan)
    for i, ((R, t), obs_uv) in enumerate(zip(poses, uv)):
        Xc = R @ X + t
        if Xc[2] <= 1e-6:
            continue
        px = to_pixel([[Xc[0], Xc[1], Xc[2]]])[0]
        if not np.all(np.isfinite(px)):
            continue
        out[i] = float(np.hypot(px[0] - obs_uv[0], px[1] - obs_uv[1]))
    return out


# ---------------------------------------------------------------- отчёт

def robust_stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {}
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return {
        "n": int(v.size),
        "median": med,
        "mad": mad,
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "p90": float(np.percentile(v, 90)),
        "max": float(v.max()),
        "cv_percent": float(100.0 * mad / med) if med else None,
    }


def make_plots(rows, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    res = np.array([r["residual_px"] for r in rows], dtype=float)
    rad = np.array([r["radius_px"] for r in rows], dtype=float)
    idx = np.array([r["frame_index"] for r in rows], dtype=float)
    ok = np.isfinite(res)
    res, rad, idx = res[ok], rad[ok], idx[ok]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax[0, 0].hist(res, bins=80)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("невязка репроекции, px")
    ax[0, 0].set_ylabel("наблюдений (log)")
    ax[0, 0].set_title("Распределение невязок (двухмодальность?)")

    ax[0, 1].scatter(rad, res, s=6, alpha=0.35)
    if rad.size > 20:
        bins = np.linspace(rad.min(), rad.max(), 12)
        cent, med = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (rad >= a) & (rad < b)
            if m.sum() >= 3:
                cent.append((a + b) / 2)
                med.append(np.median(res[m]))
        ax[0, 1].plot(cent, med, "r-o", lw=2, label="медиана по бинам")
        ax[0, 1].legend()
    ax[0, 1].set_xlabel("радиус от главной точки, px")
    ax[0, 1].set_ylabel("невязка, px")
    ax[0, 1].set_title("Рост с радиусом => дисторсия (нужен OPENCV)")

    ax[1, 0].scatter(idx, res, s=6, alpha=0.35)
    ax[1, 0].set_xlabel("номер кадра")
    ax[1, 0].set_ylabel("невязка, px")
    ax[1, 0].set_title("Кучкуется по кадрам => стабилизация / смаз")

    srt = np.sort(res)
    ax[1, 1].plot(srt, np.arange(1, srt.size + 1) / srt.size)
    for thr in (1.0, 1.5, 2.0, 3.0):
        ax[1, 1].axvline(thr, ls="--", lw=0.8, color="grey")
    ax[1, 1].set_xlabel("порог --max-reproj, px")
    ax[1, 1].set_ylabel("доля сохраняемых наблюдений")
    ax[1, 1].set_title("Цена порога отбраковки")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main

def parse_bases(items):
    out = {}
    for it in items or []:
        try:
            pair, val = it.split("=")
            a, b = pair.split("-")
            out[(int(a), int(b))] = float(val)
        except Exception:
            die("не разобрать --base '%s', нужен формат 1-2=1000" % it)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path,
                    help="папка проекта; кадры берутся из 02_frames, модель из 03_colmap/sparse/0")
    ap.add_argument("--frames", type=Path, help="папка с кадрами (переопределяет --project)")
    ap.add_argument("--sparse", type=Path, help="папка sparse-модели (переопределяет --project)")
    ap.add_argument("--out", type=Path, help="папка отчётов (по умолчанию <project>/reports)")
    ap.add_argument("--dict", default="DICT_5X5_100", help="словарь ArUco")
    ap.add_argument("--marker-size", type=float, default=100.0,
                    help="номинальная сторона маркера, мм")
    ap.add_argument("--base", action="append",
                    help="известная база между центрами: 1-2=1000 (мм), можно несколько раз")
    ap.add_argument("--min-views", type=int, default=4,
                    help="минимум наблюдений угла для триангуляции")
    ap.add_argument("--huber", type=float, default=1.5, help="порог Хьюбера в px для IRLS")
    args = ap.parse_args()

    frames = args.frames or (args.project / "02_frames" if args.project else None)
    sparse = args.sparse or (args.project / "03_colmap" / "sparse" / "0" if args.project else None)
    outdir = args.out or (args.project / "reports" if args.project else Path("."))
    if not frames or not sparse:
        die("укажите --project либо пару --frames/--sparse")
    if not Path(frames).is_dir():
        die("нет папки кадров: %s" % frames)
    if not Path(sparse).is_dir():
        die("нет sparse-модели: %s" % sparse)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print("модель:  %s" % sparse)
    print("кадры:   %s" % frames)
    rec, images = load_reconstruction(sparse)
    print("зарегистрировано изображений: %d" % len(images))

    obs, n_read = detect_all(frames, images, args.dict, (".jpg", ".jpeg", ".png", ".JPG"))
    print("прочитано кадров: %d, углов с наблюдениями: %d" % (n_read, len(obs)))
    if not obs:
        die("ни одного маркера не найдено — проверьте словарь --dict")

    cam_ops = {}
    for cam_id, cam in rec.cameras.items():
        cam_ops[cam_id] = make_camera_ops(cam)

    rows = []
    points3d = {}
    for (mid, k), lst in sorted(obs.items()):
        if len(lst) < args.min_views:
            continue
        rays, poses, uvs, names = [], [], [], []
        for name, u, v in lst:
            info = images[name]
            to_norm, to_pix, pp = cam_ops[info["camera_id"]]
            xy = to_norm([[u, v]])[0]
            if not np.all(np.isfinite(xy)):
                continue
            rays.append(xy)
            poses.append((info["R"], info["t"]))
            uvs.append((u, v))
            names.append(name)
        if len(rays) < args.min_views:
            continue
        X, res = triangulate_irls(rays, poses, uvs, cam_ops[images[names[0]]["camera_id"]][1],
                                  huber=args.huber)
        if X is None:
            continue
        points3d[(mid, k)] = X
        for name, (u, v), r in zip(names, uvs, res):
            _, _, pp = cam_ops[images[name]["camera_id"]]
            rows.append({
                "image": name,
                "frame_index": frame_index(name),
                "marker_id": mid,
                "corner": k,
                "u": round(float(u), 3),
                "v": round(float(v), 3),
                "radius_px": round(float(np.hypot(u - pp[0], v - pp[1])), 2),
                "residual_px": round(float(r), 4) if np.isfinite(r) else "",
            })

    if not rows:
        die("не удалось триангулировать ни одного угла — мало наблюдений?")

    # ---- масштабы по базам
    scales = defaultdict(list)
    detail = []
    for mid in sorted({m for m, _ in points3d}):
        c = {k: points3d[(mid, k)] for k in range(4) if (mid, k) in points3d}
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            if a in c and b in c:
                d = float(np.linalg.norm(c[a] - c[b]))
                s = args.marker_size / d
                scales["side_%.0fmm" % args.marker_size].append(s)
                detail.append({"type": "side", "marker": mid, "pair": [a, b],
                               "dist_units": d, "nominal_mm": args.marker_size,
                               "scale_mm_per_unit": s})
        for a, b in [(0, 2), (1, 3)]:
            if a in c and b in c:
                d = float(np.linalg.norm(c[a] - c[b]))
                nom = args.marker_size * math.sqrt(2)
                s = nom / d
                scales["diag_%.1fmm" % nom].append(s)
                detail.append({"type": "diagonal", "marker": mid, "pair": [a, b],
                               "dist_units": d, "nominal_mm": nom,
                               "scale_mm_per_unit": s})

    centers = {}
    for mid in sorted({m for m, _ in points3d}):
        pts = [points3d[(mid, k)] for k in range(4) if (mid, k) in points3d]
        if len(pts) == 4:
            centers[mid] = np.mean(pts, axis=0)
    for (a, b), mm in parse_bases(args.base).items():
        if a in centers and b in centers:
            d = float(np.linalg.norm(centers[a] - centers[b]))
            s = mm / d
            scales["centers_%d-%d_%.0fmm" % (a, b, mm)].append(s)
            detail.append({"type": "centers", "pair": [a, b], "dist_units": d,
                           "nominal_mm": mm, "scale_mm_per_unit": s})

    report = {
        "sparse": str(sparse),
        "frames": str(frames),
        "n_images_registered": len(images),
        "n_frames_read": n_read,
        "n_observations": len(rows),
        "residuals_px": robust_stats([r["residual_px"] for r in rows
                                      if r["residual_px"] != ""]),
        "residuals_by_marker": {
            str(m): robust_stats([r["residual_px"] for r in rows
                                  if r["marker_id"] == m and r["residual_px"] != ""])
            for m in sorted({r["marker_id"] for r in rows})
        },
        "keep_fraction_at_threshold": {
            str(t): round(float(np.mean([float(r["residual_px"]) <= t for r in rows
                                         if r["residual_px"] != ""])), 4)
            for t in (1.0, 1.5, 2.0, 3.0, 5.0)
        },
        "scale_by_base": {k: robust_stats(v) for k, v in scales.items()},
        "scale_detail": detail,
    }

    csv_path = Path(outdir) / "diag_markers.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        cols = ["image", "frame_index", "marker_id", "corner", "u", "v",
                "radius_px", "residual_px"]
        f.write(";".join(cols) + "\n")
        for r in rows:
            f.write(";".join(str(r[c]) for c in cols) + "\n")

    json_path = Path(outdir) / "diag_markers.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    png_path = Path(outdir) / "diag_markers.png"
    has_png = make_plots([r for r in rows if r["residual_px"] != ""], png_path)

    print("\nневязки, px: " + json.dumps(report["residuals_px"], ensure_ascii=False))
    print("доля сохраняемых наблюдений по порогам: "
          + json.dumps(report["keep_fraction_at_threshold"]))
    print("\nмасштаб по базам (мм/ед.):")
    for k, v in report["scale_by_base"].items():
        if v:
            print("  %-24s n=%-4d медиана %.3f  MAD %.3f  (%.3f %%)"
                  % (k, v["n"], v["median"], v["mad"], v.get("cv_percent") or 0.0))
    print("\nотчёты: %s, %s%s" % (csv_path, json_path,
                                  (", " + str(png_path)) if has_png else " (графики пропущены)"))


if __name__ == "__main__":
    main()