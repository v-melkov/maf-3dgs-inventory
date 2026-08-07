"""Этап 01 — приём исходного материала.

Инвентаризация входа, извлечение технических характеристик записи,
фиксация контрольных сумм и определение начала записи в шкале UTC.

Начало записи — критичная величина: от неё отсчитывается время каждого
кадра, а значит и присваиваемые кадру координаты. Определяется по
метаданным файла и уточняется поправкой часов камеры, полученной съёмкой
индикатора точного времени (п. 2.4.4.3 методики).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

from .core import (
    Out, Session, StageReport, fail, require_tool, run, sha256_file, tool_version,
)

STAGE = "01_raw"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".m2ts", ".insv"}


def probe(path: Path, log: Path) -> dict:
    require_tool("ffprobe", "установите ffmpeg и добавьте его в PATH")
    p = run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        log_path=log,
    )
    return json.loads(p.stdout)


def _video_stream(meta: dict) -> dict:
    for s in meta.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    fail("во входном файле нет видеопотока")
    return {}


def _parse_rate(value: str | None) -> float | None:
    if not value or value in ("0/0",):
        return None
    try:
        return float(Fraction(value))
    except Exception:  # noqa: BLE001
        return None


def _creation_time(meta: dict) -> datetime | None:
    tags = {**meta.get("format", {}).get("tags", {})}
    for s in meta.get("streams", []):
        tags.update(s.get("tags", {}))
    raw = tags.get("creation_time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def run_stage(session: Session, cfg: dict, force: bool = False) -> StageReport:
    rep = StageReport(STAGE)
    out = Out(STAGE, "Приём исходного материала")
    log = session.log_path(STAGE)
    stage_dir = session.dir(STAGE)

    videos = sorted(
        p for p in stage_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXT
    )
    if not videos:
        fail(
            f"в каталоге {stage_dir} нет видеофайлов",
            "скопируйте исходную запись обхода в этот каталог и повторите",
        )

    rep.tool_versions["ffprobe"] = tool_version(["ffprobe", "-version"])

    files: list[dict] = []
    total_duration = 0.0
    for v in videos:
        meta = probe(v, log)
        vs = _video_stream(meta)
        avg = _parse_rate(vs.get("avg_frame_rate"))
        r = _parse_rate(vs.get("r_frame_rate"))
        duration = float(meta.get("format", {}).get("duration", 0.0))
        item = {
            "name": v.name,
            "size_bytes": v.stat().st_size,
            "sha256": sha256_file(v),
            "duration_s": round(duration, 3),
            "width": vs.get("width"),
            "height": vs.get("height"),
            "codec": vs.get("codec_name"),
            "avg_frame_rate": avg,
            "r_frame_rate": r,
            "creation_time": (
                _creation_time(meta).isoformat() if _creation_time(meta) else None
            ),
            "has_data_stream": any(
                s.get("codec_type") == "data" for s in meta.get("streams", [])
            ),
        }
        files.append(item)
        rep.inputs_sha256[v.name] = item["sha256"]
        total_duration += duration

    files.sort(key=lambda i: (i["creation_time"] or "", i["name"]))

    first = files[0]
    out.kv("файлов", len(files))
    out.kv("суммарная длительность", f"{total_duration:.1f} с")
    out.kv("разрешение", f"{first['width']}\u00d7{first['height']}")
    out.kv("кодек", first["codec"])
    out.kv("частота кадров", f"{first['avg_frame_rate']:.3f}" if first["avg_frame_rate"] else "?")

    # --- проверки ---
    if total_duration < cfg["ingest"]["min_duration_s"]:
        fail(
            f"длительность записи {total_duration:.1f} с меньше допустимой "
            f"{cfg['ingest']['min_duration_s']} с",
            "для реконструкции требуется полный обход объекта",
        )
    long_side = max(first["width"] or 0, first["height"] or 0)
    if long_side < cfg["ingest"]["min_long_side"]:
        rep.warn(
            f"разрешение {long_side} px по длинной стороне ниже рекомендованного "
            f"{cfg['ingest']['min_long_side']} px (п. 2.4.2)", out,
        )
    if first["avg_frame_rate"] and first["avg_frame_rate"] < cfg["ingest"]["min_fps"]:
        rep.warn(
            f"частота кадров {first['avg_frame_rate']:.1f} ниже рекомендованной "
            f"{cfg['ingest']['min_fps']} — вероятен смаз при движении", out,
        )
    for f in files:
        if f["avg_frame_rate"] and f["r_frame_rate"]:
            if abs(f["avg_frame_rate"] - f["r_frame_rate"]) / f["r_frame_rate"] > 0.02:
                fail(
                    f"в файле {f['name']} переменная частота кадров "
                    f"(avg={f['avg_frame_rate']:.3f}, r={f['r_frame_rate']:.3f})",
                    "привязка номера кадра ко времени станет нелинейной; "
                    "перекодируйте запись с постоянной частотой: "
                    "ffmpeg -i in.mp4 -c:v libx264 -crf 16 -r 30 -an out.mp4",
                )
    if len({(f["width"], f["height"], f["codec"]) for f in files}) > 1:
        fail("файлы серии имеют разные параметры изображения",
             "обрабатывайте их как отдельные сессии")

    # --- разрывы между файлами серии ---
    for prev, cur in zip(files, files[1:]):
        if not (prev["creation_time"] and cur["creation_time"]):
            continue
        t_prev_end = datetime.fromisoformat(prev["creation_time"]) + timedelta(
            seconds=prev["duration_s"]
        )
        gap = (datetime.fromisoformat(cur["creation_time"]) - t_prev_end).total_seconds()
        if abs(gap) > cfg["ingest"]["max_series_gap_s"]:
            rep.warn(
                f"разрыв {gap:.1f} с между {prev['name']} и {cur['name']}", out
            )

    # --- начало записи в UTC ---
    clock_offset = float(cfg["time"].get("clock_offset_s", 0.0))
    explicit = cfg["time"].get("video_start_utc")
    if explicit:
        t0 = datetime.fromisoformat(str(explicit).replace("Z", "+00:00"))
        t0_source = "задано в конфигурации"
    elif first["creation_time"]:
        t0 = datetime.fromisoformat(first["creation_time"])
        t0_source = "метаданные файла (creation_time)"
    else:
        fail(
            "не удалось определить время начала записи",
            "укажите time.video_start_utc в конфигурации сессии "
            "(значение берётся по снимку индикатора точного времени)",
        )
    t0 = t0 + timedelta(seconds=clock_offset)

    out.rule()
    out.kv("начало записи (UTC)", t0.isoformat(timespec="milliseconds"))
    out.kv("источник", t0_source)
    out.kv("поправка часов", f"{clock_offset:+.3f} с")
    if not explicit and abs(clock_offset) < 1e-9:
        rep.warn(
            "поправка часов не задана: время взято из метаданных без проверки; "
            "по п. 2.4.4.3 требуется определить её съёмкой индикатора точного времени", out,
        )
    if not first["has_data_stream"]:
        rep.warn(
            "в файле нет потока данных — трек ГНСС внутри записи отсутствует, "
            "потребуется внешний источник трека", out,
        )

    rep.params = {
        "clock_offset_s": clock_offset,
        "t0_source": t0_source,
    }
    rep.metrics = {
        "n_files": len(files),
        "total_duration_s": round(total_duration, 3),
        "width": first["width"],
        "height": first["height"],
        "fps": first["avg_frame_rate"],
        "t0_utc": t0.isoformat(),
        "files": files,
    }
    rep.duration_s = out.done()
    rep.write(session, stage_dir)
    return rep
