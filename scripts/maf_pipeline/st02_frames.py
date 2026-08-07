"""Этап 02 — извлечение и отбор кадров.

Этап разделён на два прохода, и это разделение принципиально.

Проход A. Видеопоток разбирается на кадры с постоянным шагом собственным
вызовом ffmpeg. Собственный вызов, а не средства утилиты отбора, применяется
по одной причине: он даёт полный контроль над именованием, а значит
однозначное соответствие номера кадра моменту времени
    t_i = t0 + i / fps,
без которого невозможно корректное присвоение координат. Остаточная
погрешность определяется только шагом извлечения и составляет ±1/(2·fps).

Проход B. По полученному каталогу изображений выполняется отбор заданного
числа наиболее резких кадров средствами sharp-frames. Утилита применяется
исключительно к тому, для чего предназначена, — оценке резкости и отбору
с сохранением равномерности распределения по исходному материалу.

Побочное следствие разделения: исследование зависимости точности от числа
кадров (п. 3.2.7) выполняется повторным запуском только прохода B с разными
значениями --num-frames. Все наборы оказываются подмножествами одного и того
же исходного материала и различаются исключительно числом кадров, что
устраняет вариативность самого извлечения.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .core import (
    Out, Session, StageReport, ensure_free_space, fail, link_or_copy,
    require_tool, run, tool_version,
)

STAGE = "02_frames"


# ---------------------------------------------------------------------------
# Проход A — извлечение
# ---------------------------------------------------------------------------


def extract_all(session: Session, cfg: dict, raw: dict, log: Path, out: Out) -> dict:
    """Разбор видеопотока на кадры с постоянным шагом.

    Возвращает отображение «имя файла → момент съёмки (UTC)».
    """
    require_tool("ffmpeg", "установите ffmpeg и добавьте его в PATH")
    fps = float(cfg["frames"]["extract_fps"])
    quality = int(cfg["frames"]["jpeg_quality_scale"])
    all_dir = session.dir(f"{STAGE}/all")

    files = raw["metrics"]["files"]
    total_duration = raw["metrics"]["total_duration_s"]
    expected = int(total_duration * fps)

    # Оценка потребного объёма: эмпирически ~0,6 МиБ на кадр 4K при qscale 2.
    ensure_free_space(all_dir, int(expected * 0.7 * 2**20))

    clock_offset = float(raw["params"]["clock_offset_s"])
    t0_global = datetime.fromisoformat(raw["metrics"]["t0_utc"])

    times: dict[str, str] = {}
    cumulative = 0.0
    for k, f in enumerate(files):
        src = session.dir("01_raw") / f["name"]
        if f["creation_time"]:
            t0 = datetime.fromisoformat(f["creation_time"]) + timedelta(seconds=clock_offset)
        else:
            t0 = t0_global + timedelta(seconds=cumulative)
        pattern = str(all_dir / f"s{k:02d}_%06d.jpg")
        t_start = out._t0
        run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(src),
             "-vf", f"fps={fps}",
             "-qscale:v", str(quality),
             "-qmin", "1", "-qmax", "1",
             pattern],
            log_path=log,
        )
        produced = sorted(all_dir.glob(f"s{k:02d}_*.jpg"))
        for j, p in enumerate(produced):
            times[p.name] = (t0 + timedelta(seconds=j / fps)).isoformat()
        cumulative += f["duration_s"]
        out.step(f"извлечение {f['name']}", f"{len(produced)} кадров",
                 out._t0 - t_start if False else None)

    if not times:
        fail(
            "ffmpeg не извлёк ни одного кадра",
            "проверьте декодируемость записи: ffmpeg -v error -i FILE -f null -",
        )
    return times


# ---------------------------------------------------------------------------
# Проход B — отбор
# ---------------------------------------------------------------------------


def _resolve_selection(tmp_dir: Path, all_dir: Path) -> list[str]:
    """Восстанавливает список отобранных ИСХОДНЫХ имён.

    Схема selected_metadata.json утилитой не документирована и может
    измениться между версиями, поэтому разбор выполняется по нескольким
    вариантам, а при неудаче — сопоставлением по основе имени файла.
    """
    meta_path = tmp_dir / "selected_metadata.json"
    known = {p.name for p in all_dir.iterdir() if p.is_file()}
    stems = {p.stem: p.name for p in all_dir.iterdir() if p.is_file()}

    names: list[str] = []
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
        except json.JSONDecodeError:
            meta = None
        records = None
        if isinstance(meta, list):
            records = meta
        elif isinstance(meta, dict):
            for key in ("selected_frames", "selected", "frames", "images", "results"):
                if isinstance(meta.get(key), list):
                    records = meta[key]
                    break
        for rec in records or []:
            if isinstance(rec, str):
                cand = Path(rec).name
            elif isinstance(rec, dict):
                cand = None
                for key in ("source_path", "source", "input_path", "original_path",
                            "path", "filename", "file", "name"):
                    if rec.get(key):
                        cand = Path(str(rec[key])).name
                        break
            else:
                cand = None
            if not cand:
                continue
            if cand in known:
                names.append(cand)
            elif Path(cand).stem in stems:
                names.append(stems[Path(cand).stem])

    if not names:
        # Запасной путь: сопоставление выходных файлов с исходными по основе имени.
        for p in sorted(tmp_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if p.name in known:
                names.append(p.name)
            elif p.stem in stems:
                names.append(stems[p.stem])

    if not names:
        keys = ""
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text("utf-8"))
                keys = ", ".join(m.keys()) if isinstance(m, dict) else f"список из {len(m)}"
            except Exception:  # noqa: BLE001
                keys = "не разобран"
        fail(
            "не удалось сопоставить отобранные кадры с исходными",
            f"структура selected_metadata.json: [{keys}] — "
            "проверьте формат вывода установленной версии sharp-frames "
            "и при необходимости дополните _resolve_selection()",
        )
    return sorted(set(names))


def select_frames(session: Session, cfg: dict, log: Path, out: Out) -> tuple[list[str], dict]:
    all_dir = session.dir(f"{STAGE}/all")
    tmp_dir = session.dir(f"{STAGE}/_select_tmp")
    for p in tmp_dir.iterdir():
        p.unlink() if p.is_file() else shutil.rmtree(p)

    n = int(cfg["frames"]["num_frames"])
    cmd = [
        "sharp-frames", str(all_dir), str(tmp_dir),
        "--selection-method", cfg["frames"]["selection_method"],
        "--num-frames", str(n),
        "--min-buffer", str(cfg["frames"]["min_buffer"]),
        "--format", "jpg",
        "--force-overwrite",
    ]
    run(cmd, log_path=log, echo=True)

    names = _resolve_selection(tmp_dir, all_dir)
    meta_path = tmp_dir / "selected_metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
        except json.JSONDecodeError:
            meta = {}
        shutil.copy2(meta_path, session.dir(STAGE) / "selected_metadata.json")

    sel_dir = session.dir(f"{STAGE}/selected")
    for p in sel_dir.iterdir():
        if p.is_file():
            p.unlink()
    for name in names:
        link_or_copy(all_dir / name, sel_dir / name)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return names, meta


# ---------------------------------------------------------------------------
# Временные метки
# ---------------------------------------------------------------------------


def write_exif_times(session: Session, names: list[str], times: dict[str, str],
                     log: Path) -> None:
    """Запись времени съёмки в EXIF отобранных кадров.

    Время пишется в UTC с явным указанием нулевого смещения: это исключает
    любые преобразования часовых поясов на этапе присвоения координат.
    Дробная часть записывается отдельным тегом — она используется при
    интерполяции положения между отметками трека.
    """
    require_tool("exiftool", "загрузите exiftool с exiftool.org и добавьте в PATH")
    sel_dir = session.dir(f"{STAGE}/selected")
    args: list[str] = []
    for name in names:
        t = datetime.fromisoformat(times[name]).astimezone(timezone.utc)
        args += [
            f"-DateTimeOriginal={t.strftime('%Y:%m:%d %H:%M:%S')}",
            f"-SubSecTimeOriginal={t.microsecond // 1000:03d}",
            "-OffsetTimeOriginal=+00:00",
            "-OffsetTimeDigitized=+00:00",
            f"-ImageDescription={session.id}/{name}",
            "-overwrite_original",
            str(sel_dir / name),
            "-execute",
        ]
    args.append("-common_args")
    run(["exiftool", "-@", "-"], log_path=log, stdin_text="\n".join(args),
        ok_codes=(0, 1))


# ---------------------------------------------------------------------------
# Контроль
# ---------------------------------------------------------------------------


def contact_sheet(session: Session, names: list[str], cols: int = 6,
                  rows: int = 4) -> Path | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    sel_dir = session.dir(f"{STAGE}/selected")
    qc_dir = session.dir(f"{STAGE}/qc")
    k = cols * rows
    if not names:
        return None
    step = max(1, len(names) // k)
    picked = names[::step][:k]
    thumb = (320, 240)
    sheet = Image.new("RGB", (cols * thumb[0], rows * thumb[1]), "black")
    for i, name in enumerate(picked):
        with Image.open(sel_dir / name) as im:
            im = im.convert("RGB")
            im.thumbnail(thumb)
            sheet.paste(im, ((i % cols) * thumb[0], (i // cols) * thumb[1]))
    p = qc_dir / "contact_sheet.jpg"
    sheet.save(p, quality=88)
    return p


# ---------------------------------------------------------------------------
# Этап
# ---------------------------------------------------------------------------


def run_stage(session: Session, cfg: dict, force: bool = False) -> StageReport:
    session.require_stage("01_raw")
    rep = StageReport(STAGE)
    out = Out(STAGE, "Извлечение и отбор кадров")
    log = session.log_path(STAGE)
    stage_dir = session.dir(STAGE)
    raw = session.report_of("01_raw")

    rep.tool_versions["ffmpeg"] = tool_version(["ffmpeg", "-version"])
    rep.tool_versions["sharp-frames"] = tool_version(["sharp-frames", "--version"])
    rep.tool_versions["exiftool"] = tool_version(["exiftool", "-ver"])

    all_dir = session.dir(f"{STAGE}/all")
    existing = sorted(p.name for p in all_dir.glob("*.jpg"))
    times_path = stage_dir / "frame_times.json"

    if existing and times_path.exists() and not force:
        times = json.loads(times_path.read_text("utf-8"))
        out.step("извлечение", f"{len(existing)} кадров (уже выполнено)")
    else:
        times = extract_all(session, cfg, raw, log, out)
        times_path.write_text(json.dumps(times, ensure_ascii=False, indent=1), "utf-8")

    names, sf_meta = select_frames(session, cfg, log, out)
    out.step("отбор по резкости", f"{len(names)} из {len(times)}")

    write_exif_times(session, names, times, log)
    out.step("запись времени в EXIF", f"{len(names)} кадров")

    # --- frames.json: единая таблица кадров рабочего набора ---
    scores = _scores_by_name(sf_meta)
    frames = [
        {
            "name": n,
            "t_utc": times[n],
            "sharpness": scores.get(n),
        }
        for n in names
    ]
    (stage_dir / "frames.json").write_text(
        json.dumps(frames, ensure_ascii=False, indent=1), "utf-8"
    )

    # --- проверки ---
    fps = float(cfg["frames"]["extract_fps"])
    target = int(cfg["frames"]["num_frames"])
    if len(names) < 0.9 * target:
        rep.warn(
            f"отобрано {len(names)} кадров при запрошенных {target}", out
        )
    if len(names) < cfg["frames"]["min_frames_hard"]:
        fail(
            f"отобрано всего {len(names)} кадров",
            "для реконструкции требуется существенно больше; "
            "проверьте длительность записи и параметр frames.extract_fps",
        )
    vals = [s for s in scores.values() if isinstance(s, (int, float))]
    if vals:
        vals_sorted = sorted(vals)
        median = vals_sorted[len(vals_sorted) // 2]
        weak = [n for n, s in scores.items()
                if n in set(names) and isinstance(s, (int, float)) and s < 0.3 * median]
        if weak:
            rep.warn(
                f"{len(weak)} кадров с резкостью ниже 0,3 медианы "
                f"(первые: {', '.join(weak[:5])})", out
            )

    sheet = contact_sheet(session, names)
    if sheet:
        rep.qc_images.append(str(sheet.relative_to(session.root)))

    out.rule()
    out.kv("шаг извлечения", f"{1 / fps:.3f} с")
    out.kv("квантование времени", f"\u00b1{1 / (2 * fps):.3f} с")
    out.kv("кадров всего / отобрано", f"{len(times)} / {len(names)}")
    out.qc([Path(q) for q in rep.qc_images])

    rep.params = {
        "extract_fps": fps,
        "num_frames_requested": target,
        "selection_method": cfg["frames"]["selection_method"],
        "min_buffer": cfg["frames"]["min_buffer"],
        "jpeg_quality_scale": cfg["frames"]["jpeg_quality_scale"],
    }
    rep.metrics = {
        "n_extracted": len(times),
        "n_selected": len(names),
        "time_quantization_s": round(1 / (2 * fps), 4),
        "sharpness_available": bool(vals),
    }
    rep.duration_s = out.done()
    rep.write(session, stage_dir)
    return rep


def _scores_by_name(meta) -> dict[str, float]:
    res: dict[str, float] = {}
    records = None
    if isinstance(meta, list):
        records = meta
    elif isinstance(meta, dict):
        for key in ("selected_frames", "selected", "frames", "images", "results"):
            if isinstance(meta.get(key), list):
                records = meta[key]
                break
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        name = None
        for key in ("source_path", "source", "path", "filename", "file", "name"):
            if rec.get(key):
                name = Path(str(rec[key])).name
                break
        score = None
        for key in ("sharpness", "score", "sharpness_score", "combined_score"):
            if isinstance(rec.get(key), (int, float)):
                score = float(rec[key])
                break
        if name and score is not None:
            res[name] = score
    return res
