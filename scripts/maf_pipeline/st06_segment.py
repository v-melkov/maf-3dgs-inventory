"""Этап 06 — двумерная семантическая сегментация объекта.

Единственный этап, требующий участия оператора. Причина изложена в методике:
текстовый запрос детектора с открытым словарём возвращает все объекты,
соответствующие описанию, а в сцене благоустройства объекты учёта расположены
близко и часто попадают в один кадр. Автоматический выбор по наибольшему
доверительному значению или площади ненадёжен и невоспроизводим, поэтому
выбор делает оператор — однократно, на опорном кадре.

Сделанный выбор сохраняется в object.json и при повторном запуске
используется без запроса, чем обеспечивается воспроизводимость обработки.

Детектор берётся из библиотеки transformers, а не из эталонного репозитория:
реализация в transformers не содержит нативных расширений CUDA, требующих
компиляции, что существенно упрощает развёртывание под Windows при
неизменной модели и лицензии.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .core import Out, Session, StageReport, fail, link_or_copy

STAGE = "06_objects"


# ---------------------------------------------------------------------------
# Детектирование
# ---------------------------------------------------------------------------


def detect_candidates(image_paths: list[Path], prompt: str, cfg: dict) -> list[dict]:
    """Возвращает список рамок-кандидатов по всем просмотренным кадрам."""
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = cfg["segment"]["detector_model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    text = prompt.strip()
    if not text.endswith("."):
        text += "."

    found: list[dict] = []
    for p in image_paths:
        image = Image.open(p).convert("RGB")
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=cfg["segment"]["box_threshold"],
            text_threshold=cfg["segment"]["text_threshold"],
            target_sizes=[image.size[::-1]],
        )[0]
        w, h = image.size
        for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
            x0, y0, x1, y1 = box
            area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0)) / (w * h)
            touches = x0 <= 2 or y0 <= 2 or x1 >= w - 2 or y1 >= h - 2
            found.append({
                "image": p.name, "box": [x0, y0, x1, y1],
                "score": float(score), "area_frac": area, "touches_border": touches,
            })
    return found


def pick_reference(found: list[dict], cfg: dict) -> str:
    """Опорный кадр — тот, где объект виден наиболее полно.

    Требования: доверительное значение выше порога, рамка не касается границ
    кадра (иначе объект попал в кадр не целиком), максимальная площадь.
    """
    good = [f for f in found
            if not f["touches_border"] and f["score"] >= cfg["segment"]["ref_min_score"]]
    pool = good or found
    if not pool:
        fail("детектор не обнаружил объект ни на одном из просмотренных кадров",
             "уточните текстовый запрос (segment.prompt) или понизьте "
             "segment.box_threshold")
    best = max(pool, key=lambda f: f["area_frac"])
    return best["image"]


def draw_candidates(image_path: Path, boxes: list[dict], dst: Path) -> Path:
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for i, b in enumerate(boxes):
        x0, y0, x1, y1 = b["box"]
        d.rectangle([x0, y0, x1, y1], outline=(255, 80, 0), width=4)
        label = f"{i}  {b['score']:.2f}"
        d.rectangle([x0, max(0, y0 - 26), x0 + 9 * len(label) + 8, y0], fill=(255, 80, 0))
        d.text((x0 + 4, max(0, y0 - 23)), label, fill=(0, 0, 0))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=92)
    return dst


# ---------------------------------------------------------------------------
# Распространение маски
# ---------------------------------------------------------------------------


def propagate(session: Session, obj_dir: Path, names: list[str], ref_name: str,
              box: list[float], cfg: dict, out: Out) -> dict[str, float]:
    """Распространение маски по последовательности в обе стороны от опорного кадра."""
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    images = session.dir("02_frames/selected")
    # Предиктор последовательности ожидает каталог кадров, именованных
    # порядковыми номерами; формируем его жёсткими ссылками.
    seq = obj_dir / "_seq"
    if seq.exists():
        shutil.rmtree(seq, ignore_errors=True)
    seq.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        link_or_copy(images / n, seq / f"{i}.jpg")
    ref_idx = names.index(ref_name)

    predictor = build_sam2_video_predictor(
        cfg["segment"]["sam2_config"], cfg["segment"]["sam2_checkpoint"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    masks_dir = obj_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    areas: dict[str, float] = {}

    with torch.inference_mode():
        state = predictor.init_state(video_path=str(seq))
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=ref_idx, obj_id=1,
            box=np.array(box, dtype=np.float32),
        )
        for reverse in (False, True):
            for idx, obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=ref_idx, reverse=reverse
            ):
                mask = (logits[0] > 0.0).cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                name = names[idx]
                arr = (mask.astype("uint8") * 255)
                Image.fromarray(arr).save(masks_dir / (Path(name).stem + ".png"))
                areas[name] = float(mask.sum()) / mask.size

    shutil.rmtree(seq, ignore_errors=True)
    out.step("распространение маски", f"{len(areas)} кадров")
    return areas


# ---------------------------------------------------------------------------
# Контроль качества масок
# ---------------------------------------------------------------------------


def check_masks(session: Session, obj_dir: Path, names: list[str],
                areas: dict[str, float], cfg: dict, rep: StageReport,
                out: Out) -> list[str]:
    """Реализация проверок п. 2.4.6.4. Возвращает список отбракованных кадров."""
    import numpy as np
    from PIL import Image

    masks_dir = obj_dir / "masks"
    rejected: list[str] = []
    reasons: dict[str, list[str]] = {}

    def reject(name: str, why: str) -> None:
        reasons.setdefault(name, []).append(why)
        if name not in rejected:
            rejected.append(name)

    for name in names:
        mp = masks_dir / (Path(name).stem + ".png")
        if not mp.exists() or areas.get(name, 0.0) <= 0:
            reject(name, "маска не сформирована")
            continue
        if areas[name] < cfg["segment"]["min_area_frac"]:
            reject(name, "площадь маски пренебрежимо мала")

    present = [n for n in names if n not in rejected]
    for a, b in zip(present, present[1:]):
        fa, fb = areas[a], areas[b]
        if fa > 0 and abs(fb - fa) / fa > cfg["segment"]["max_area_jump"]:
            reject(b, f"скачок площади {100 * (fb - fa) / fa:+.0f} %")

    border_hits = 0
    for name in present:
        arr = np.array(Image.open(masks_dir / (Path(name).stem + ".png")))
        if arr[0].any() or arr[-1].any() or arr[:, 0].any() or arr[:, -1].any():
            border_hits += 1

    share_rejected = len(rejected) / len(names) if names else 1.0
    out.kv("отбраковано кадров", f"{len(rejected)} ({100 * share_rejected:.1f} %)")
    out.kv("маска касается границ", f"{border_hits} кадров")

    (obj_dir / "rejected.json").write_text(
        json.dumps(reasons, ensure_ascii=False, indent=1), "utf-8"
    )
    if share_rejected > cfg["segment"]["max_rejected_share"]:
        fail(
            f"отбраковано {100 * share_rejected:.1f} % кадров — сегментация неустойчива",
            "выберите другой опорный кадр (segment.reference_image) "
            "либо уточните текстовый запрос",
        )
    if rejected:
        rep.warn(f"{len(rejected)} кадров исключены из обучающей выборки", out)
    if border_hits > 0.5 * len(present):
        rep.warn("более половины кадров содержат объект не целиком", out)
    return rejected


def make_qc(session: Session, obj_dir: Path, names: list[str],
            areas: dict[str, float], n: int = 12) -> list[Path]:
    from PIL import Image
    qc = obj_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    images = session.dir("02_frames/selected")
    masks = obj_dir / "masks"

    step = max(1, len(names) // n)
    picked = names[::step][:n]
    tiles = []
    for name in picked:
        mp = masks / (Path(name).stem + ".png")
        if not mp.exists():
            continue
        im = Image.open(images / name).convert("RGB")
        mk = Image.open(mp).convert("L").resize(im.size)
        overlay = Image.new("RGB", im.size, (255, 60, 0))
        im = Image.composite(Image.blend(im, overlay, 0.45), im, mk)
        im.thumbnail((420, 320))
        tiles.append(im)
    paths = []
    if tiles:
        cols, rows = 4, (len(tiles) + 3) // 4
        w, h = tiles[0].size
        sheet = Image.new("RGB", (cols * w, rows * h), "black")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * w, (i // cols) * h))
        p = qc / "overlay_sheet.jpg"
        sheet.save(p, quality=90)
        paths.append(p)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.plot(range(len(names)), [areas.get(n_, 0.0) for n_ in names], lw=1.0)
        ax.set_xlabel("номер кадра в последовательности")
        ax.set_ylabel("доля площади кадра")
        ax.set_title("Площадь маски объекта")
        ax.grid(alpha=0.3)
        p2 = qc / "area_curve.png"
        fig.savefig(p2, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(p2)
    except ImportError:
        pass
    return paths


# ---------------------------------------------------------------------------
# Этап
# ---------------------------------------------------------------------------


def run_stage(session: Session, cfg: dict, object_id: str, prompt: str | None = None,
              force: bool = False, assume_yes: bool = False) -> StageReport:
    session.require_stage("05_sfm")
    rep = StageReport(f"{STAGE}/{object_id}")
    out = Out("06_objects", f"Сегментация объекта: {object_id}")
    obj_dir = session.object_dir(object_id)
    meta_path = obj_dir / "object.json"

    # --- обучающая выборка: только зарегистрированные кадры ---
    centers = json.loads(
        (session.dir("05_sfm") / "camera_centers.json").read_text("utf-8")
    )
    names = sorted(centers)
    if not names:
        fail("нет зарегистрированных кадров", "проверьте результат этапа 05")

    meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
    if force:
        meta.pop("reference_image", None)
        meta.pop("box", None)

    prompt = prompt or meta.get("prompt") or cfg["segment"].get("prompt")
    if not prompt:
        fail("не задан текстовый запрос",
             "укажите --prompt или segment.prompt в конфигурации")
    out.kv("запрос", prompt)

    # --- выбор объекта (интерактивный, однократный) ---
    if "box" not in meta:
        images = session.dir("02_frames/selected")
        stride = max(1, len(names) // cfg["segment"]["detect_samples"])
        sample = [images / n for n in names[::stride]]
        out.step("детектирование", f"просмотрено {len(sample)} кадров")
        found = detect_candidates(sample, prompt, cfg)
        ref_name = meta.get("reference_image") or pick_reference(found, cfg)
        on_ref = sorted([f for f in found if f["image"] == ref_name],
                        key=lambda f: -f["area_frac"])
        out.kv("опорный кадр", ref_name)
        out.kv("обнаружено рамок", len(on_ref))

        if len(on_ref) == 1 or assume_yes:
            chosen = on_ref[0]
        else:
            preview = draw_candidates(images / ref_name, on_ref,
                                      obj_dir / "qc" / "candidates.jpg")
            print(f"  на опорном кадре обнаружено несколько объектов, "
                  f"соответствующих запросу")
            print(f"  откройте: {preview}")
            for i, b in enumerate(on_ref):
                print(f"      {i}: доверие {b['score']:.2f}, "
                      f"площадь {100 * b['area_frac']:.1f} % кадра"
                      + (", касается границы" if b["touches_border"] else ""))
            raw = input("  номер объекта учёта: ").strip()
            try:
                chosen = on_ref[int(raw)]
            except (ValueError, IndexError):
                fail(f"неверный номер: {raw!r}")

        meta.update({
            "object_id": object_id,
            "prompt": prompt,
            "reference_image": ref_name,
            "box": chosen["box"],
            "score": chosen["score"],
            "detector_model": cfg["segment"]["detector_model"],
        })
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    else:
        out.kv("опорный кадр", meta["reference_image"] + " (из object.json)")

    # --- распространение и контроль ---
    areas = propagate(session, obj_dir, names, meta["reference_image"],
                      meta["box"], cfg, out)
    rejected = check_masks(session, obj_dir, names, areas, cfg, rep, out)
    kept = [n for n in names if n not in rejected]

    qc_paths = make_qc(session, obj_dir, kept, areas)
    rep.qc_images = [str(p.relative_to(session.root)) for p in qc_paths]
    out.qc(qc_paths, require_confirm=True)

    (obj_dir / "train_frames.json").write_text(
        json.dumps(kept, ensure_ascii=False, indent=1), "utf-8"
    )

    rep.params = {
        "prompt": prompt,
        "reference_image": meta["reference_image"],
        "detector_model": cfg["segment"]["detector_model"],
        "sam2_checkpoint": cfg["segment"]["sam2_checkpoint"],
    }
    rep.metrics = {
        "n_frames": len(names),
        "n_rejected": len(rejected),
        "share_rejected": round(len(rejected) / len(names), 4),
        "n_train_frames": len(kept),
        "mean_area_frac": round(sum(areas.values()) / max(len(areas), 1), 5),
    }
    rep.duration_s = out.done()
    rep.write(session, obj_dir)
    return rep
