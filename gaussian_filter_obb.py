#!/usr/bin/env python3
"""
Фильтрация артефактных гауссианов сегментированного объекта и построение
ориентированного ограничивающего параллелепипеда (OBB).

Этап пайплайна: выполняется ПОСЛЕ переноса 2D-масок в 3D (получения подмножества
гауссианов объекта) и ДО снятия габаритных характеристик.

Вход:  .ply файл 3DGS с гауссианами, отнесёнными к объекту
Выход: отфильтрованный .ply + габариты L x W x H + отчёт по этапам фильтрации

Зависимости: numpy, scipy, scikit-learn, plyfile
    pip install numpy scipy scikit-learn plyfile
"""

import argparse
import json
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------------------
# Чтение / запись 3DGS .ply
# ----------------------------------------------------------------------------

def load_gaussians(path):
    """Читает .ply формата 3DGS.

    Возвращает словарь с сырыми полями и производными величинами:
      xyz     — центры гауссианов, м (в единицах модели)
      opacity — непрозрачность после сигмоиды, [0, 1]
      scale   — масштабы по трём осям после exp, в единицах модели
    В формате 3DGS opacity и scale хранятся в лог-пространстве:
    активация — sigmoid(opacity) и exp(scale) соответственно.
    """
    ply = PlyData.read(path)
    v = ply["vertex"].data

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    scale = np.exp(np.stack(
        [v["scale_0"], v["scale_1"], v["scale_2"]], axis=1
    ).astype(np.float64))

    return {"xyz": xyz, "opacity": opacity, "scale": scale, "raw": v}


def save_gaussians(path, raw, mask):
    """Сохраняет подмножество гауссианов, сохраняя все исходные поля."""
    el = PlyElement.describe(raw[mask], "vertex")
    PlyData([el]).write(path)


# ----------------------------------------------------------------------------
# Ступень 1. Фильтр по непрозрачности
# ----------------------------------------------------------------------------

def filter_by_opacity(g, thr=0.10):
    """Отсекает почти прозрачные гауссианы.

    Обоснование: гауссианы с малой альфой вносят пренебрежимо малый вклад в
    рендеринг (и потому не штрафуются при обучении и не видны на метриках
    PSNR/mIoU), но при построении OBB по экстремумам учитываются наравне
    с остальными.
    """
    return g["opacity"] >= thr


# ----------------------------------------------------------------------------
# Ступень 2. Фильтр по размеру гауссиана
# ----------------------------------------------------------------------------

def filter_by_scale(g, max_scale_m=None, percentile=99.0):
    """Отсекает аномально크 крупные («размазанные») гауссианы.

    Такие гауссианы типичны для областей с недостаточным угловым покрытием
    и для моделирования бликов на глянцевых поверхностях: один гауссиан
    растягивается на десятки сантиметров, искажая границу объекта.

    max_scale_m — абсолютный порог в единицах модели (если модель уже
    отмасштабирована — в метрах). Если не задан, используется процентиль.
    """
    smax = g["scale"].max(axis=1)
    thr = max_scale_m if max_scale_m is not None else np.percentile(smax, percentile)
    return smax <= thr


# ----------------------------------------------------------------------------
# Ступень 3. Статистическое удаление выбросов (SOR)
# ----------------------------------------------------------------------------

def filter_statistical_outliers(xyz, k=20, std_ratio=2.0):
    """Statistical Outlier Removal.

    Для каждой точки считается среднее расстояние до k ближайших соседей.
    Точки, у которых оно превышает mean + std_ratio * std по всему облаку,
    признаются выбросами. Убирает одиночные «floater»-гауссианы, висящие
    в стороне от объекта.
    """
    k = min(k, len(xyz) - 1)
    if k < 1:
        return np.ones(len(xyz), dtype=bool)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(xyz)
    dist, _ = nn.kneighbors(xyz)
    mean_dist = dist[:, 1:].mean(axis=1)  # без самой точки

    thr = mean_dist.mean() + std_ratio * mean_dist.std()
    return mean_dist <= thr


# ----------------------------------------------------------------------------
# Ступень 4. Кластеризация DBSCAN — оставляем основной кластер
# ----------------------------------------------------------------------------

def filter_dbscan(xyz, eps=None, min_samples=10, eps_knn_percentile=90):
    """Оставляет только крупнейший связный кластер.

    Убирает целые группы артефактных гауссианов, пространственно отделённые
    от объекта (фрагменты фона, «протёкшие» через маску).

    Если eps не задан, оценивается автоматически по распределению расстояний
    до k-го соседа — стандартная эвристика подбора eps для DBSCAN.
    """
    if eps is None:
        k = min(min_samples, len(xyz) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(xyz)
        dist, _ = nn.kneighbors(xyz)
        eps = np.percentile(dist[:, -1], eps_knn_percentile)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz)

    valid = labels[labels >= 0]
    if len(valid) == 0:
        return np.ones(len(xyz), dtype=bool), eps

    main = np.bincount(valid).argmax()
    return labels == main, eps


# ----------------------------------------------------------------------------
# Построение OBB
# ----------------------------------------------------------------------------

def compute_obb(xyz, weights=None, trim_percentile=0.5, vertical_axis=2):
    """Ориентированный ограничивающий параллелепипед через PCA.

    weights          — веса точек (рекомендуется передавать opacity: более
                       «плотные» гауссианы сильнее влияют на ориентацию осей)
    trim_percentile  — процент отсечения по краям вдоль каждой оси; защищает
                       габарит от единичных остаточных выбросов. При 0 берутся
                       абсолютные min/max.
    vertical_axis    — индекс мировой вертикальной оси. Высота МАФ считается
                       вдоль неё, а PCA применяется только в горизонтальной
                       плоскости: объекты благоустройства стоят на земле, и
                       наклонный «оптимальный» бокс для них лишён смысла.

    Возвращает габариты (длина, ширина, высота), центр, матрицу осей.
    """
    if weights is None:
        weights = np.ones(len(xyz))

    horiz = [i for i in range(3) if i != vertical_axis]
    pts_h = xyz[:, horiz]

    # Взвешенный PCA в горизонтальной плоскости
    mean_h = np.average(pts_h, axis=0, weights=weights)
    centered = pts_h - mean_h
    cov = np.cov(centered.T, aweights=weights)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    axes_h = eigvecs[:, order]  # 2x2: главная ось = длина

    proj = centered @ axes_h

    def extent(a):
        if trim_percentile > 0:
            lo = np.percentile(a, trim_percentile)
            hi = np.percentile(a, 100 - trim_percentile)
        else:
            lo, hi = a.min(), a.max()
        return lo, hi, hi - lo

    lo_l, hi_l, length = extent(proj[:, 0])
    lo_w, hi_w, width = extent(proj[:, 1])
    lo_v, hi_v, height = extent(xyz[:, vertical_axis])

    center_h = mean_h + axes_h @ np.array([(lo_l + hi_l) / 2, (lo_w + hi_w) / 2])
    center = np.zeros(3)
    center[horiz] = center_h
    center[vertical_axis] = (lo_v + hi_v) / 2

    axes = np.eye(3)
    axes[np.ix_(horiz, horiz)] = axes_h

    return {
        "length": length,
        "width": width,
        "height": height,
        "center": center,
        "axes": axes,
    }


# ----------------------------------------------------------------------------
# Основной конвейер
# ----------------------------------------------------------------------------

def run(path_in, path_out=None, scale_factor=1.0, params=None, verbose=True):
    p = {
        "opacity_thr": 0.10,
        "scale_percentile": 99.0,
        "sor_k": 20,
        "sor_std_ratio": 2.0,
        "dbscan_min_samples": 10,
        "dbscan_eps": None,
        "trim_percentile": 0.5,
        "vertical_axis": 2,
    }
    if params:
        p.update(params)

    g = load_gaussians(path_in)
    n0 = len(g["xyz"])
    keep = np.ones(n0, dtype=bool)
    report = {"n_input": n0, "params": {k: v for k, v in p.items()}, "stages": []}

    def log(name, new_keep):
        removed = int(keep.sum() - new_keep.sum())
        report["stages"].append({
            "stage": name,
            "removed": removed,
            "remaining": int(new_keep.sum()),
        })
        if verbose:
            print(f"{name:<28} удалено: {removed:>7}   осталось: {new_keep.sum():>7}")

    # Ступень 1 — непрозрачность
    m = filter_by_opacity(g, p["opacity_thr"])
    new_keep = keep & m
    log("1. Непрозрачность", new_keep)
    keep = new_keep

    # Ступень 2 — размер гауссиана
    m = filter_by_scale(g, percentile=p["scale_percentile"])
    new_keep = keep & m
    log("2. Размер гауссиана", new_keep)
    keep = new_keep

    # Ступень 3 — статистические выбросы
    idx = np.where(keep)[0]
    m_sub = filter_statistical_outliers(
        g["xyz"][idx], k=p["sor_k"], std_ratio=p["sor_std_ratio"]
    )
    new_keep = np.zeros(n0, dtype=bool)
    new_keep[idx[m_sub]] = True
    log("3. Стат. выбросы (SOR)", new_keep)
    keep = new_keep

    # Ступень 4 — DBSCAN, основной кластер
    idx = np.where(keep)[0]
    m_sub, eps_used = filter_dbscan(
        g["xyz"][idx],
        eps=p["dbscan_eps"],
        min_samples=p["dbscan_min_samples"],
    )
    new_keep = np.zeros(n0, dtype=bool)
    new_keep[idx[m_sub]] = True
    log("4. DBSCAN (осн. кластер)", new_keep)
    keep = new_keep
    report["dbscan_eps_used"] = float(eps_used)

    # Габариты до и после фильтрации — для иллюстрации в работе
    obb_raw = compute_obb(
        g["xyz"], g["opacity"], trim_percentile=0.0,
        vertical_axis=p["vertical_axis"]
    )
    obb = compute_obb(
        g["xyz"][keep], g["opacity"][keep],
        trim_percentile=p["trim_percentile"],
        vertical_axis=p["vertical_axis"],
    )

    def dims(o):
        return {
            "L_mm": round(o["length"] * scale_factor * 1000, 1),
            "W_mm": round(o["width"] * scale_factor * 1000, 1),
            "H_mm": round(o["height"] * scale_factor * 1000, 1),
        }

    report["n_output"] = int(keep.sum())
    report["removed_total_pct"] = round(100 * (1 - keep.sum() / n0), 2)
    report["obb_before_filtering"] = dims(obb_raw)
    report["obb_after_filtering"] = dims(obb)
    report["scale_factor"] = scale_factor

    if verbose:
        print(f"\nВсего удалено: {report['removed_total_pct']}% гауссианов")
        print(f"Габариты до фильтрации : {report['obb_before_filtering']}")
        print(f"Габариты после         : {report['obb_after_filtering']}")

    if path_out:
        save_gaussians(path_out, g["raw"], keep)

    return report


def ablation(path_in, scale_factor=1.0, thresholds=None):
    """Прогон при разных порогах непрозрачности — материал для таблицы ablation."""
    thresholds = thresholds or [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    rows = []
    for thr in thresholds:
        r = run(path_in, None, scale_factor,
                params={"opacity_thr": thr}, verbose=False)
        rows.append({
            "opacity_thr": thr,
            "remaining_pct": round(100 * r["n_output"] / r["n_input"], 1),
            **r["obb_after_filtering"],
        })

    print(f"{'порог α':>9} {'осталось,%':>11} {'L, мм':>9} {'W, мм':>9} {'H, мм':>9}")
    for r in rows:
        print(f"{r['opacity_thr']:>9.2f} {r['remaining_pct']:>11.1f} "
              f"{r['L_mm']:>9.1f} {r['W_mm']:>9.1f} {r['H_mm']:>9.1f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="входной .ply с гауссианами объекта")
    ap.add_argument("-o", "--output", help="выходной .ply (отфильтрованный)")
    ap.add_argument("-s", "--scale", type=float, default=1.0,
                    help="масштабный коэффициент из калибровки по эталону")
    ap.add_argument("--opacity-thr", type=float, default=0.10)
    ap.add_argument("--sor-std", type=float, default=2.0)
    ap.add_argument("--dbscan-min-samples", type=int, default=10)
    ap.add_argument("--vertical-axis", type=int, default=2,
                    help="индекс вертикальной оси мира (0=x, 1=y, 2=z)")
    ap.add_argument("--report", help="путь для сохранения отчёта в JSON")
    ap.add_argument("--ablation", action="store_true",
                    help="прогон по сетке порогов непрозрачности")
    args = ap.parse_args()

    if args.ablation:
        ablation(args.input, args.scale)
    else:
        rep = run(args.input, args.output, args.scale, params={
            "opacity_thr": args.opacity_thr,
            "sor_std_ratio": args.sor_std,
            "dbscan_min_samples": args.dbscan_min_samples,
            "vertical_axis": args.vertical_axis,
        })
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=2)