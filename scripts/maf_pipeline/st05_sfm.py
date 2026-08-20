"""Этап 05 — оценка поз камер.

Выполняется по НЕмаскированным кадрам: алгоритм восстановления структуры
по движению использует характерные точки всей сцены, включая фон, а линейный
эталон расположен вне объекта учёта и на маскированных кадрах отсутствовал бы
(п. 2.4.6.5 методики).

Наиболее опасная нештатная ситуация этапа — распад реконструкции на несколько
подмоделей. Формально обработка завершается успешно, но подмодели соответствуют
разным участкам сцены и объединению не подлежат. Проверка выполняется явно,
продолжение требует подтверждения оператора.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

from .core import (
    Out, Session, StageReport, fail, require_tool, run, tool_version,
)

STAGE = "05_sfm"


# ---------------------------------------------------------------------------
# Разбор результатов
# ---------------------------------------------------------------------------


def _model_dirs(sparse: Path) -> list[Path]:
    return sorted(
        d for d in sparse.iterdir()
        if d.is_dir() and (d / "images.bin").exists() or (d / "images.txt").exists()
    ) if sparse.exists() else []


def _analyze(model_dir: Path, log: Path) -> dict:
    proc = run(["colmap", "model_analyzer", "--path", str(model_dir)],
               log_path=log, check=False)
    text = (proc.stdout or "") + (proc.stderr or "")
    res: dict = {}
    patterns = {
        "n_cameras": r"Cameras:\s*(\d+)",
        "n_images": r"Images:\s*(\d+)",
        "n_points": r"Points:\s*(\d+)",
        "n_observations": r"Observations:\s*(\d+)",
        "mean_track_length": r"Mean track length:\s*([\d.]+)",
        "mean_obs_per_image": r"Mean observations per image:\s*([\d.]+)",
        "mean_reproj_error": r"Mean reprojection error:\s*([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            res[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    return res


def _camera_centers(model_dir: Path, log: Path, work: Path) -> dict[str, list[float]]:
    """Центры фотографирования в системе координат модели.

    Извлекаются через текстовое представление модели: центр камеры
    C = -R^T * t, где R задаётся кватернионом из images.txt.
    """
    txt = work / "sparse_txt"
    txt.mkdir(parents=True, exist_ok=True)
    run(["colmap", "model_converter", "--input_path", str(model_dir),
         "--output_path", str(txt), "--output_type", "TXT"], log_path=log)

    centers: dict[str, list[float]] = {}
    lines = (txt / "images.txt").read_text("utf-8").splitlines()
    data_lines = [ln for ln in lines if ln and not ln.startswith("#")]
    for ln in data_lines[::2]:  # каждая вторая строка — точки, они не нужны
        parts = ln.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        r = [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
        t = (tx, ty, tz)
        centers[name] = [-sum(r[i][j] * t[i] for i in range(3)) for j in range(3)]
    return centers


def _plot_cameras(session: Session, centers: dict[str, list[float]]) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    qc = session.dir(f"{STAGE}/qc")
    pts = list(centers.values())
    if not pts:
        return []
    # Проекция на плоскость двух главных компонент: ориентация системы
    # координат COLMAP произвольна, и «план» в ней заранее не определён.
    m = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
    c = [[p[i] - m[i] for i in range(3)] for p in pts]
    cov = [[sum(a[i] * a[j] for a in c) / len(c) for j in range(3)] for i in range(3)]
    try:
        import numpy as np
        w, v = np.linalg.eigh(np.array(cov))
        axes = v[:, np.argsort(w)[::-1][:2]]
        proj = np.array(c) @ axes
        x, y = proj[:, 0], proj[:, 1]
    except Exception:  # noqa: BLE001
        x = [a[0] for a in c]
        y = [a[1] for a in c]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, "-", lw=0.6, alpha=0.6)
    ax.scatter(x, y, s=10)
    ax.set_aspect("equal")
    ax.set_title("Центры фотографирования\n(проекция на плоскость главных компонент)")
    ax.set_xlabel("единицы модели")
    ax.grid(alpha=0.3)
    p = qc / "cameras_plan.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return [p]


# ---------------------------------------------------------------------------
# Этап
# ---------------------------------------------------------------------------


def run_stage(session: Session, cfg: dict, force: bool = False,
              assume_yes: bool = False) -> StageReport:
    session.require_stage("02_frames")
    rep = StageReport(STAGE)
    out = Out(STAGE, "Оценка поз камер", "COLMAP")
    log = session.log_path(STAGE)
    stage_dir = session.dir(STAGE)

    require_tool("colmap", "установите COLMAP со сборкой под CUDA и добавьте в PATH")
    rep.tool_versions["colmap"] = tool_version(["colmap", "-h"])

    images = session.dir("02_frames/selected")
    n_images = len(list(images.glob("*.jpg")))
    db = stage_dir / "database.db"
    sparse = session.dir(f"{STAGE}/sparse")

    if force and db.exists():
        db.unlink()
        shutil.rmtree(sparse, ignore_errors=True)
        sparse.mkdir(parents=True, exist_ok=True)

    camera_model = cfg["sfm"]["camera_model"]
    use_gpu = "1" if cfg["sfm"]["use_gpu"] else "0"

    out.kv("вход", f"{n_images} кадров")
    out.kv("модель камеры", camera_model)
    out.rule()

    # --- извлечение признаков ---
    run(["colmap", "feature_extractor",
         "--database_path", str(db),
         "--image_path", str(images),
         "--ImageReader.camera_model", camera_model,
         "--ImageReader.single_camera", "1",
         "--SiftExtraction.use_gpu", use_gpu,
         "--SiftExtraction.max_image_size", str(cfg["sfm"]["max_image_size"])],
        log_path=log)
    out.step("извлечение признаков", f"{n_images}/{n_images}")

    # --- сопоставление ---
    matcher = cfg["sfm"]["matcher"]
    if matcher == "auto":
        matcher = "exhaustive" if n_images <= cfg["sfm"]["exhaustive_limit"] else "sequential"
    if matcher == "exhaustive":
        run(["colmap", "exhaustive_matcher",
             "--database_path", str(db),
             "--SiftMatching.use_gpu", use_gpu,
             "--SiftMatching.max_num_matches", str(cfg["sfm"]["max_num_matches"])],
            log_path=log)
    else:
        cmd = ["colmap", "sequential_matcher",
               "--database_path", str(db),
               "--SiftMatching.use_gpu", use_gpu,
               "--SequentialMatching.overlap", str(cfg["sfm"]["sequential_overlap"])]
        vocab = cfg["sfm"].get("vocab_tree_path")
        if vocab:
            cmd += ["--SequentialMatching.loop_detection", "1",
                    "--SequentialMatching.vocab_tree_path", str(vocab)]
        run(cmd, log_path=log)
    out.step(f"сопоставление ({matcher})", "выполнено")

    # --- реконструкция ---
    run(["colmap", "mapper",
         "--database_path", str(db),
         "--image_path", str(images),
         "--output_path", str(sparse)],
        log_path=log)

    models = _model_dirs(sparse)
    if not models:
        fail(
            "реконструкция не построена",
            "недостаточное перекрытие между кадрами либо однородная сцена; "
            "увеличьте frames.num_frames и повторите этапы 02, 04, 05",
        )

    stats = {d.name: _analyze(d, log) for d in models}
    main = max(models, key=lambda d: stats[d.name].get("n_images", 0))
    main_stats = stats[main.name]
    registered = int(main_stats.get("n_images", 0))
    share = registered / n_images if n_images else 0.0

    out.step("реконструкция", f"подмоделей: {len(models)}")
    out.rule()
    out.kv("основная подмодель", main.name)
    out.kv("зарегистрировано", f"{registered}/{n_images} ({100 * share:.1f} %)")
    out.kv("точек", main_stats.get("n_points", "?"))
    out.kv("ошибка репроекции", f"{main_stats.get('mean_reproj_error', float('nan')):.2f} px")
    out.kv("средняя длина трека", f"{main_stats.get('mean_track_length', float('nan')):.1f}")

    # --- проверки ---
    if len(models) > 1:
        dist = ", ".join(f"{d.name}: {stats[d.name].get('n_images', 0)}" for d in models)
        rep.warn(
            f"реконструкция распалась на {len(models)} подмодели ({dist}); "
            f"дальше используется только {main.name}", out,
        )
        print("      Подмодели соответствуют разным участкам сцены и объединению")
        print("      не подлежат. Типичные причины: быстрое движение, участок")
        print("      с однородным фоном, недостаточное число кадров.")
        if not assume_yes:
            ans = input("      продолжить с основной подмоделью? [y/N] ").strip().lower()
            if ans not in ("y", "yes", "д", "да"):
                fail("обработка прервана оператором",
                     "увеличьте frames.num_frames либо переснимите объект")

    if share < cfg["sfm"]["min_registered_fail"]:
        fail(
            f"зарегистрировано лишь {100 * share:.1f} % кадров",
            "реконструкция ненадёжна; увеличьте число кадров или проверьте, "
            "не была ли включена цифровая стабилизация при съёмке",
        )
    if share < cfg["sfm"]["min_registered_warn"]:
        rep.warn(f"не зарегистрировано {n_images - registered} кадров", out)
    if main_stats.get("mean_reproj_error", 0) > cfg["sfm"]["max_reproj_error"]:
        rep.warn(f"ошибка репроекции {main_stats['mean_reproj_error']:.2f} px "
                 f"выше допустимой {cfg['sfm']['max_reproj_error']} px", out)
    if main_stats.get("mean_track_length", 99) < cfg["sfm"]["min_track_length"]:
        rep.warn(f"средняя длина трека точки {main_stats['mean_track_length']:.1f} "
                 f"указывает на слабое перекрытие кадров", out)

    centers = _camera_centers(main, log, stage_dir)
    (stage_dir / "camera_centers.json").write_text(
        json.dumps(centers, ensure_ascii=False, indent=1), "utf-8"
    )
    qc_paths = _plot_cameras(session, centers)
    rep.qc_images = [str(p.relative_to(session.root)) for p in qc_paths]
    out.qc(qc_paths, require_confirm=True)

    unreg = sorted(set(p.name for p in images.glob("*.jpg")) - set(centers))
    if unreg:
        print(f"  \u26a0 не зарегистрированы: {', '.join(unreg[:6])}"
              + (f" и ещё {len(unreg) - 6}" if len(unreg) > 6 else ""))

    rep.params = {
        "camera_model": camera_model,
        "matcher": matcher,
        "max_image_size": cfg["sfm"]["max_image_size"],
    }
    rep.metrics = {
        "n_input_images": n_images,
        "n_submodels": len(models),
        "main_model": main.name,
        "n_registered": registered,
        "share_registered": round(share, 4),
        "submodel_stats": stats,
        "unregistered": unreg,
    }
    rep.duration_s = out.done()
    rep.write(session, stage_dir)
    return rep
