"""Этап 07 — обучение 3DGS-модели объекта.

На вход подаётся набор в структуре COLMAP: разрежённая модель из этапа 05,
изображения обучающей выборки и соответствующие им маски. Маски применяются
только здесь (п. 2.4.6.5): на этапе оценки поз камер они не используются.

Каталог набора собирается жёсткими ссылками, а не копированием: это позволяет
пересобрать его под иную схему размещения масок без переизвлечения кадров.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .core import (
    Out, Session, StageReport, fail, link_or_copy, run, tool_version, which,
)

STAGE = "07_models"


def build_dataset(session: Session, cfg: dict, object_id: str, out: Out) -> Path:
    """Формирование каталога набора данных для обучения."""
    obj_dir = session.object_dir(object_id, create=False)
    kept = json.loads((obj_dir / "train_frames.json").read_text("utf-8"))
    if not kept:
        fail("обучающая выборка пуста", "проверьте результат этапа 06")

    ds = session.dir(f"{STAGE}/{object_id}/dataset")
    images_dir = ds / cfg["train"]["images_dir_name"]
    masks_dir = ds / cfg["train"]["masks_dir_name"]
    for d in (images_dir, masks_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    src_images = session.dir("02_frames/selected")
    src_masks = obj_dir / "masks"
    for name in kept:
        link_or_copy(src_images / name, images_dir / name)
        mask = src_masks / (Path(name).stem + ".png")
        if mask.exists():
            # Имя маски приводится к имени изображения: среды обучения
            # сопоставляют их по имени, а расширение может учитываться
            # или не учитываться в зависимости от версии.
            link_or_copy(mask, masks_dir / (Path(name).stem + ".png"))
            link_or_copy(mask, masks_dir / (name + ".png"))

    # Разрежённая модель: обучение ведётся в той же системе координат,
    # в которой определяется масштабный коэффициент по эталону.
    sfm_report = session.report_of("05_sfm")
    main = sfm_report["metrics"]["main_model"]
    src_sparse = session.dir("05_sfm/sparse") / main
    dst_sparse = ds / "sparse" / "0"
    shutil.rmtree(dst_sparse, ignore_errors=True)
    dst_sparse.mkdir(parents=True, exist_ok=True)
    for f in src_sparse.iterdir():
        if f.is_file():
            link_or_copy(f, dst_sparse / f.name)

    out.step("набор данных", f"{len(kept)} кадров и масок")
    return ds


def run_stage(session: Session, cfg: dict, object_id: str,
              force: bool = False) -> StageReport:
    rep = StageReport(f"{STAGE}/{object_id}")
    out = Out("07_models", f"Обучение 3DGS-модели: {object_id}", "LichtFeld Studio")
    log = session.log_path(f"07_{object_id}")
    model_dir = session.dir(f"{STAGE}/{object_id}")

    obj_dir = session.object_dir(object_id, create=False)
    if not (obj_dir / "train_frames.json").exists():
        fail(f"объект {object_id} не сегментирован",
             f"выполните: python run.py segment {session.root} --object {object_id}")

    binary = cfg["train"].get("binary") or which("lichtfeld-studio") or \
        which("LichtFeld-Studio")
    if not binary:
        fail("не найдена LichtFeld Studio",
             "укажите полный путь к исполняемому файлу в train.binary")
    rep.tool_versions["lichtfeld-studio"] = tool_version([binary, "--version"])

    # --- проверка вычислительных ресурсов ---
    try:
        import torch
        if not torch.cuda.is_available():
            fail("CUDA недоступна", "обучение 3DGS выполняется только на GPU")
        free, total = torch.cuda.mem_get_info()
        out.kv("видеопамять", f"{free / 2**30:.1f} из {total / 2**30:.1f} ГиБ свободно")
        if free < cfg["train"]["min_free_vram_gib"] * 2**30:
            fail(
                f"свободно {free / 2**30:.1f} ГиБ видеопамяти при требуемых "
                f"{cfg['train']['min_free_vram_gib']} ГиБ",
                "закройте приложения, использующие GPU, либо понизьте "
                "train.resize_factor / train.max_cap",
            )
    except ImportError:
        rep.warn("PyTorch не установлен — объём видеопамяти не проверен", out)

    kept = json.loads((obj_dir / "train_frames.json").read_text("utf-8"))
    out.kv("кадров обучающей выборки", len(kept))
    if len(kept) < cfg["train"]["min_frames_warn"]:
        rep.warn(f"{len(kept)} кадров — заведомо низкое качество реконструкции", out)

    ds = build_dataset(session, cfg, object_id, out)

    # --- команда обучения ---
    camera_model = session.report_of("05_sfm")["params"]["camera_model"]
    cmd = [
        binary,
        "-d", str(ds),
        "-o", str(model_dir),
        "--output-name", "splat",
        "--headless", "--train", "--no-splash",
        "--images", cfg["train"]["images_dir_name"],
        "--mask-mode", "segment",
        "--iter", str(cfg["train"]["iterations"]),
        "--strategy", cfg["train"]["strategy"],
        "--sh-degree", str(cfg["train"]["sh_degree"]),
        "--log-file", str(model_dir / "train.log"),
        "--log-level", "info",
    ]
    if cfg["train"].get("max_cap"):
        cmd += ["--max-cap", str(cfg["train"]["max_cap"])]
    if cfg["train"].get("resize_factor"):
        cmd += ["-r", str(cfg["train"]["resize_factor"])]
    if cfg["train"].get("tile_mode"):
        cmd += ["--tile-mode", str(cfg["train"]["tile_mode"])]
    if cfg["train"].get("eval", True):
        cmd += ["--eval", "--save-eval-images"]
    if camera_model not in ("PINHOLE", "SIMPLE_PINHOLE"):
        # Нелинейная проекция: растеризация некорректна, требуется
        # трассировка с учётом дисторсии.
        cmd += ["--gut"]
        out.kv("режим рендеринга", f"3DGUT (модель камеры {camera_model})")

    run(cmd, log_path=log, echo=True)

    # --- контроль результата ---
    plys = sorted(model_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime)
    if not plys:
        fail("модель не создана: выходной .ply не найден",
             f"см. журнал обучения: {model_dir / 'train.log'}")
    ply = plys[-1]
    n_gauss = _count_vertices(ply)

    out.rule()
    out.kv("модель", ply.name)
    out.kv("размер файла", f"{ply.stat().st_size / 2**20:.1f} МиБ")
    out.kv("гауссианов", f"{n_gauss:,}".replace(",", "\u202f") if n_gauss else "?")

    if n_gauss and n_gauss < cfg["train"]["min_gaussians"]:
        rep.warn(
            f"всего {n_gauss} гауссианов — вероятная причина: маски инвертированы; "
            "проверьте изображения оценки и при подтверждении добавьте "
            "--invert-masks в train.extra_args", out,
        )

    out.qc([model_dir / "train.log"], require_confirm=True)
    print("      просмотрите модель перед переходом к масштабированию:")
    print(f"      {binary} -v {ply}")

    rep.params = {
        "iterations": cfg["train"]["iterations"],
        "strategy": cfg["train"]["strategy"],
        "mask_mode": "segment",
        "gut": camera_model not in ("PINHOLE", "SIMPLE_PINHOLE"),
        "resize_factor": cfg["train"].get("resize_factor"),
        "max_cap": cfg["train"].get("max_cap"),
    }
    rep.metrics = {
        "n_train_frames": len(kept),
        "ply": str(ply.relative_to(session.root)),
        "ply_size_bytes": ply.stat().st_size,
        "n_gaussians": n_gauss,
    }
    rep.duration_s = out.done()
    rep.write(session, model_dir)
    return rep


def _count_vertices(ply: Path) -> int | None:
    """Число гауссианов из заголовка .ply без чтения тела файла."""
    try:
        with open(ply, "rb") as f:
            for _ in range(64):
                line = f.readline().decode("ascii", "replace").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:  # noqa: BLE001
        return None
    return None
