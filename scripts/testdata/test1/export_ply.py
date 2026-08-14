#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_ply.py — выгрузка разрежённого облака COLMAP в PLY для MeshLab.

По умолчанию пишет облако в единицах COLMAP: измеренное в MeshLab расстояние
надо будет умножить на масштабный коэффициент из st08.

С ключом --scale облако сразу масштабируется в миллиметры, и MeshLab
показывает готовые миллиметры. Для главы 3 удобнее второй вариант —
меньше ручной арифметики и меньше шансов перепутать коэффициенты сессий.

Фильтры --min-track и --max-error убирают слабые точки: они почти всегда
шумовые и мешают целиться в край объекта.

Примеры:
  python export_ply.py 03_colmap/sparse/0_txt reports/sparse.ply
  python export_ply.py 03_colmap/sparse/0_txt reports/sparse_mm.ply --scale 333.3218
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sparse", type=Path, help="папка sparse-модели (bin или txt)")
    ap.add_argument("out", type=Path, help="выходной .ply")
    ap.add_argument("--scale", type=float, default=None,
                    help="масштабный коэффициент мм/ед.; с ним координаты будут в мм")
    ap.add_argument("--min-track", type=int, default=3,
                    help="минимальная длина трека точки (по умолчанию 3)")
    ap.add_argument("--max-error", type=float, default=2.0,
                    help="максимальная невязка точки, px (по умолчанию 2.0)")
    ap.add_argument("--cameras", action="store_true",
                    help="добавить центры камер отдельными красными точками")
    args = ap.parse_args()

    try:
        import pycolmap
    except ImportError:
        print("ОШИБКА: не установлен pycolmap", file=sys.stderr)
        sys.exit(1)

    rec = pycolmap.Reconstruction(str(args.sparse))
    xyz, rgb = [], []
    n_all = 0
    for p in rec.points3D.values():
        n_all += 1
        track = getattr(p, "track", None)
        tlen = track.length() if track is not None and hasattr(track, "length") else 999
        err = float(getattr(p, "error", 0.0) or 0.0)
        if tlen < args.min_track or err > args.max_error:
            continue
        xyz.append(np.asarray(p.xyz, dtype=float))
        c = np.asarray(getattr(p, "color", [200, 200, 200])).astype(int).ravel()[:3]
        rgb.append(c)

    if not xyz:
        print("ОШИБКА: после фильтрации не осталось точек — ослабьте --min-track/--max-error",
              file=sys.stderr)
        sys.exit(1)

    xyz = np.vstack(xyz)
    rgb = np.vstack(rgb)

    if args.cameras:
        cams = []
        for img in rec.images.values():
            rigid = getattr(img, "cam_from_world", None)
            if callable(rigid):
                rigid = rigid()
            if rigid is None:
                continue
            M = rigid.matrix() if hasattr(rigid, "matrix") else None
            if M is None:
                continue
            M = np.asarray(M, dtype=float)
            R, t = M[:3, :3], M[:3, 3]
            cams.append(-R.T @ t)
        if cams:
            xyz = np.vstack([xyz, np.vstack(cams)])
            rgb = np.vstack([rgb, np.tile([255, 0, 0], (len(cams), 1))])
            print("добавлено центров камер: %d" % len(cams))

    unit_ply, unit_ru = "colmap_units", "ед. COLMAP"
    if args.scale:
        xyz = xyz * args.scale
        unit_ply, unit_ru = "mm", "мм"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("comment units: %s\n" % unit_ply)
        if args.scale:
            f.write("comment scale_mm_per_unit: %.6f\n" % args.scale)
        f.write("element vertex %d\n" % len(xyz))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write("%.6f %.6f %.6f %d %d %d\n" % (x, y, z, r, g, b))

    ext = xyz.max(axis=0) - xyz.min(axis=0)
    print("точек в модели: %d, записано: %d" % (n_all, len(xyz)))
    print("габарит всего облака: %.3f x %.3f x %.3f %s" % (ext[0], ext[1], ext[2], unit_ru))
    print("файл: %s" % args.out)


if __name__ == "__main__":
    main()