#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап 10. Геопривязка модели.

Преобразование подобия (7 параметров) между системой координат модели и
локальной топоцентрической системой обхода: центры фотографирования из
этапа 05 сопоставляются с координатами тех же кадров из этапа 04.

Масштаб из решения НЕ извлекается. Он фиксируется значением, полученным
на этапе 08 по линейному эталону, и в оценке участвует как известная
величина. Обоснование — п. 2.5: точность бытового ГНСС-приёмника на
масштабе одного обхода (единицы метров) на порядки хуже, чем точность
эталонной базы, и решение с семью свободными параметрами унесло бы
метрику модели вслед за шумом трека.

Оценка робастная: RANSAC по тройкам соответствий, затем IRLS с функцией
Хьюбера по консенсусу. Это отсекает кадры, которым геотегирование
присвоило координату интерполяцией через разрыв трека.

РЕЖИМ ЗАГЛУШКИ. Если разброс координат кадров вырожден (все кадры несут
одну отметку — так бывает, когда в клипе записана единственная точка ГНСС
вместо непрерывного трека), задача не имеет решения: определить поворот
не по чему. Этап не отказывает, а записывает вырожденное преобразование —
поворот единичный, перенос совмещает начало координат модели с этой
единственной отметкой, масштаб из этапа 08 — и помечает результат как
`mode: "stub"`. Дальнейшие этапы получают формально корректный
`transform.json` и работают, но карточка объекта будет содержать отметку
о непроверенной геопривязке, а в сводные таблицы главы 3 такая сессия
входит только по метрике габаритов.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .core import Out, Session, StageError, StageReport

STAGE = "10_georef"

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


# ---------------------------------------------------------------------------
# Геодезия
# ---------------------------------------------------------------------------


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Топоцентрические координаты относительно точки отсчёта, метры.

    Радиусы кривизны берутся в точке отсчёта: обход одного объекта имеет
    протяжённость десятки метров, на такой базе изменением кривизны
    пренебрегают.
    """
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    alt = np.asarray(alt, dtype=float)
    lat0r, lon0r = math.radians(lat0), math.radians(lon0)

    s = math.sin(lat0r)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s * s)          # первый вертикал
    m = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s * s) ** 1.5  # меридиан

    east = (lon - lon0r) * (n + alt0) * math.cos(lat0r)
    north = (lat - lat0r) * (m + alt0)
    up = alt - alt0
    return np.stack([east, north, up], axis=1)


# ---------------------------------------------------------------------------
# Преобразование подобия с фиксированным масштабом
# ---------------------------------------------------------------------------


def fit_rigid(src: np.ndarray, dst: np.ndarray, scale: float,
              weights: np.ndarray | None = None):
    """Ищет R и t в задаче dst ≈ scale * R @ src + t.

    Масштаб задан, поэтому свободны только поворот и перенос: это метод
    Умеямы без оценки масштабного множителя.
    """
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float)
    w = w / max(w.sum(), 1e-12)

    src_c = (w[:, None] * src).sum(axis=0)
    dst_c = (w[:, None] * dst).sum(axis=0)
    p = src - src_c
    q = dst - dst_c

    h = (w[:, None] * p).T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T

    t = dst_c - scale * (r @ src_c)
    return r, t


def residuals_m(src, dst, r, t, scale):
    pred = scale * (src @ r.T) + t
    return np.linalg.norm(pred - dst, axis=1)


def fit_robust(src, dst, scale, threshold_m, iterations, huber_m, rng):
    """RANSAC по тройкам, затем IRLS по консенсусу."""
    n = len(src)
    best_inliers = None
    best_count = -1

    if n < 3:
        raise StageError(
            f"для оценки геопривязки нужно не менее 3 соответствий, есть {n}",
            "проверьте долю кадров с координатами в отчёте этапа 04",
        )

    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        try:
            r, t = fit_rigid(src[idx], dst[idx], scale)
        except np.linalg.LinAlgError:
            continue
        res = residuals_m(src, dst, r, t, scale)
        inliers = res <= threshold_m
        if inliers.sum() > best_count:
            best_count = int(inliers.sum())
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        raise StageError(
            "RANSAC не нашёл согласованного подмножества соответствий",
            "вероятная причина — сопоставление кадров и трека по времени; "
            "сверьте time.clock_offset_s и track.timezone_offset_h",
        )

    # IRLS с функцией Хьюбера по всем точкам, стартуя с консенсуса
    r, t = fit_rigid(src[best_inliers], dst[best_inliers], scale)
    for _ in range(10):
        res = residuals_m(src, dst, r, t, scale)
        w = np.where(res <= huber_m, 1.0, huber_m / np.maximum(res, 1e-9))
        r, t = fit_rigid(src, dst, scale, weights=w)

    res = residuals_m(src, dst, r, t, scale)
    return r, t, res, best_inliers


# ---------------------------------------------------------------------------
# Входные данные
# ---------------------------------------------------------------------------


def load_correspondences(session: Session) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Центры фотографирования из этапа 05 и координаты из этапа 04."""
    centers_path = session.dir("05_sfm") / "camera_centers.json"
    if not centers_path.exists():
        raise StageError(
            f"нет файла центров фотографирования: {centers_path}",
            "перезапустите этап 05: python run.py sfm <сессия> --force",
        )
    centers = json.loads(centers_path.read_text("utf-8"))

    coords_path = session.dir("04_geotag") / "image_coords.json"
    if not coords_path.exists():
        raise StageError(
            f"нет таблицы координат кадров: {coords_path}",
            "этап 04 должен записывать image_coords.json: "
            "{имя кадра: [широта, долгота, высота]}",
        )
    coords = json.loads(coords_path.read_text("utf-8"))

    names, src, geo = [], [], []
    for name, xyz in centers.items():
        c = coords.get(name)
        if not c:
            continue
        names.append(name)
        src.append([float(v) for v in xyz])
        geo.append([float(c[0]), float(c[1]), float(c[2]) if len(c) > 2 else 0.0])

    if not names:
        raise StageError(
            "ни один кадр реконструкции не имеет координат",
            "имена кадров в camera_centers.json и image_coords.json должны совпадать",
        )
    return names, np.asarray(src, dtype=float), np.asarray(geo, dtype=float)


def scale_from_stage08(session: Session) -> float:
    rep = session.report_of("08_scale")
    s = rep.get("metrics", {}).get("scale_mm_per_unit")
    if not s:
        raise StageError(
            "в отчёте этапа 08 нет масштабного коэффициента",
            "выполните: python run.py scale <сессия>",
        )
    return float(s) / 1000.0  # мм/ед. -> м/ед.


# ---------------------------------------------------------------------------
# Этап
# ---------------------------------------------------------------------------


def run_stage(session: Session, cfg: dict, force: bool = False) -> StageReport:
    session.require_stage("05_sfm")
    session.require_stage("08_scale")

    rep = StageReport(STAGE)
    out = Out(STAGE, "Геопривязка модели")
    stage_dir = session.dir(STAGE)
    gcfg = cfg.get("georef", {})

    scale_m = scale_from_stage08(session)
    names, src, geo = load_correspondences(session)
    out.kv("соответствий", len(names))
    out.kv("масштаб из этапа 08", f"{scale_m * 1000:.4f} мм/ед.")

    lat0, lon0, alt0 = float(geo[:, 0].mean()), float(geo[:, 1].mean()), float(geo[:, 2].mean())
    dst = geodetic_to_enu(geo[:, 0], geo[:, 1], geo[:, 2], lat0, lon0, alt0)
    span = float(np.linalg.norm(dst[:, :2].max(axis=0) - dst[:, :2].min(axis=0)))
    out.kv("разброс отметок в плане", f"{span:.2f} м")

    min_span = float(gcfg.get("min_span_m", 2.0))
    stub = span < min_span

    if stub:
        # Вырожденная задача: поворот определить не по чему.
        r = np.eye(3)
        t = dst.mean(axis=0) - scale_m * (r @ src.mean(axis=0))
        res = residuals_m(src, dst, r, t, scale_m)
        inliers = np.ones(len(src), dtype=bool)
        rep.warn(
            f"разброс координат кадров {span:.2f} м меньше {min_span:.1f} м — "
            "геопривязка не решается, записано вырожденное преобразование "
            "(режим заглушки, поворот единичный)", out,
        )
        rep.warn(
            "модель не ориентирована по сторонам света; результаты этой сессии "
            "пригодны для оценки габаритов, но не для позиционных выводов", out,
        )
    else:
        rng = np.random.default_rng(int(gcfg.get("seed", 0)))
        r, t, res, inliers = fit_robust(
            src, dst, scale_m,
            threshold_m=float(gcfg.get("ransac_threshold_m", 3.0)),
            iterations=int(gcfg.get("ransac_iterations", 2000)),
            huber_m=float(gcfg.get("huber_m", 2.0)),
            rng=rng,
        )

    rmse = float(np.sqrt(np.mean(res[inliers] ** 2)))
    out.kv("доля инлаеров", f"{inliers.mean() * 100:.1f} %")
    out.kv("СКО невязки", f"{rmse:.2f} м")

    warn_rmse = float(gcfg.get("rmse_warn_m", 5.0))
    if not stub and rmse > warn_rmse:
        rep.warn(
            f"СКО геопривязки {rmse:.2f} м превышает {warn_rmse:.1f} м — "
            "проверьте качество трека и сопоставление по времени", out,
        )

    transform = {
        "mode": "stub" if stub else "similarity",
        "scale_m_per_unit": scale_m,
        "scale_source": "08_scale",
        "rotation": r.tolist(),
        "translation": t.tolist(),
        "origin_wgs84": {"lat": lat0, "lon": lon0, "alt": alt0},
        "frame": "ENU (восток, север, вверх) относительно origin_wgs84",
        "n_correspondences": len(names),
        "n_inliers": int(inliers.sum()),
        "rmse_m": rmse,
    }
    (stage_dir / "transform.json").write_text(
        json.dumps(transform, ensure_ascii=False, indent=2), "utf-8"
    )

    per_image = {
        n: {"residual_m": round(float(v), 3), "inlier": bool(k)}
        for n, v, k in zip(names, res, inliers)
    }
    (stage_dir / "residuals.json").write_text(
        json.dumps(per_image, ensure_ascii=False, indent=2), "utf-8"
    )

    rep.params = {
        "min_span_m": min_span,
        "ransac_threshold_m": gcfg.get("ransac_threshold_m", 3.0),
        "huber_m": gcfg.get("huber_m", 2.0),
        "scale_fixed_from": "08_scale",
    }
    rep.metrics = {
        "mode": transform["mode"],
        "n_correspondences": len(names),
        "n_inliers": int(inliers.sum()),
        "inlier_share": round(float(inliers.mean()), 4),
        "span_m": round(span, 2),
        "georef_rmse_m": round(rmse, 3),
        "scale_mm_per_unit": round(scale_m * 1000, 4),
    }
    rep.duration_s = out.done()
    rep.write(session, stage_dir)
    return rep