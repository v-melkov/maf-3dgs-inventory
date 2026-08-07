"""Этап 03 — подготовка трека ГНСС.

Этап устроен как диспетчер источников: методика не привязана к конкретному
устройству регистрации, поэтому трек может поступать из телеметрии
видеофайла, из готового файла GPX, записанного автономным регистратором,
либо из журнала NMEA. Все варианты приводятся к единому GPX, после чего
контроль качества выполняется одинаково независимо от происхождения данных.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .core import Out, Session, StageReport, fail, run, tool_version

STAGE = "03_track"


# ---------------------------------------------------------------------------
# Источники
# ---------------------------------------------------------------------------


def _from_video(session: Session, cfg: dict, log: Path) -> Path:
    raw_dir = session.dir("01_raw")
    raw = session.report_of("01_raw")
    videos = [raw_dir / f["name"] for f in raw["metrics"]["files"]]
    dst = session.dir(STAGE) / "track_raw.gpx"
    cmd = ["pyosmogps",
           "--timezone-offset", str(cfg["track"]["timezone_offset_h"]),
           "--frequency", str(cfg["track"]["frequency_hz"]),
           "--resampling-method", cfg["track"]["resampling_method"],
           "extract", *[str(v) for v in videos], str(dst)]
    proc = run(cmd, log_path=log, check=False, echo=True)
    if proc.returncode != 0 or not dst.exists():
        fail(
            "не удалось извлечь трек из видеофайла",
            "утилита поддерживает ограниченный перечень моделей камер; "
            "при несовместимости используйте внешний регистратор и укажите "
            "track.source: gpx с путём к файлу в track.path",
        )
    return dst


def _from_gpx(session: Session, cfg: dict) -> Path:
    src = cfg["track"].get("path")
    if not src:
        fail("не задан track.path при track.source: gpx")
    src = Path(src)
    if not src.is_absolute():
        src = session.root / src
    if not src.exists():
        fail(f"файл трека не найден: {src}")
    dst = session.dir(STAGE) / "track_raw.gpx"
    shutil.copy2(src, dst)
    return dst


def _from_nmea(session: Session, cfg: dict, log: Path) -> Path:
    """Преобразование журнала NMEA средствами gpsbabel, если он доступен."""
    src = Path(cfg["track"].get("path", ""))
    if not src.is_absolute():
        src = session.root / src
    if not src.exists():
        fail(f"журнал NMEA не найден: {src}")
    dst = session.dir(STAGE) / "track_raw.gpx"
    run(["gpsbabel", "-i", "nmea", "-f", str(src), "-o", "gpx", "-F", str(dst)],
        log_path=log)
    return dst


# ---------------------------------------------------------------------------
# Разбор и контроль
# ---------------------------------------------------------------------------


def read_points(path: Path) -> list[dict]:
    import gpxpy

    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    pts: list[dict] = []
    for trk in gpx.tracks:
        for seg_i, seg in enumerate(trk.segments):
            for p in seg.points:
                if p.time is None:
                    continue
                pts.append({
                    "t": p.time.astimezone(timezone.utc),
                    "lat": p.latitude,
                    "lon": p.longitude,
                    "ele": p.elevation,
                    "seg": seg_i,
                })
    pts.sort(key=lambda x: x["t"])
    return pts


def haversine(a: dict, b: dict) -> float:
    r = 6371008.8
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dp = p2 - p1
    dl = math.radians(b["lon"] - a["lon"])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def plot_track(session: Session, pts: list[dict]) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    qc = session.dir(f"{STAGE}/qc")
    lat0 = sum(p["lat"] for p in pts) / len(pts)
    kx = 111320 * math.cos(math.radians(lat0))
    ky = 110540
    x = [(p["lon"] - pts[0]["lon"]) * kx for p in pts]
    y = [(p["lat"] - pts[0]["lat"]) * ky for p in pts]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, "-", lw=0.8)
    ax.scatter(x, y, s=4)
    ax.scatter([x[0]], [y[0]], s=60, marker="o", label="начало")
    ax.scatter([x[-1]], [y[-1]], s=60, marker="s", label="конец")
    ax.set_aspect("equal")
    ax.set_xlabel("восток, м")
    ax.set_ylabel("север, м")
    ax.set_title("Трек обхода в плане")
    ax.grid(alpha=0.3)
    ax.legend()
    p1 = qc / "track_plan.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)

    out_paths = [p1]
    if any(p["ele"] is not None for p in pts):
        t0 = pts[0]["t"]
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot([(p["t"] - t0).total_seconds() for p in pts],
                [p["ele"] for p in pts], "-", lw=0.9)
        ax.set_xlabel("время от начала, с")
        ax.set_ylabel("высота, м")
        ax.set_title("Профиль высоты")
        ax.grid(alpha=0.3)
        p2 = qc / "track_alt.png"
        fig.savefig(p2, dpi=130, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(p2)
    return out_paths


# ---------------------------------------------------------------------------
# Этап
# ---------------------------------------------------------------------------


def run_stage(session: Session, cfg: dict, force: bool = False) -> StageReport:
    session.require_stage("02_frames")
    rep = StageReport(STAGE)
    out = Out(STAGE, "Подготовка трека ГНСС")
    log = session.log_path(STAGE)
    stage_dir = session.dir(STAGE)

    source = cfg["track"]["source"]
    out.kv("источник", source)
    if source == "video":
        rep.tool_versions["pyosmogps"] = tool_version(["pyosmogps", "--help"])
        raw_gpx = _from_video(session, cfg, log)
    elif source == "gpx":
        raw_gpx = _from_gpx(session, cfg)
    elif source == "nmea":
        raw_gpx = _from_nmea(session, cfg, log)
    else:
        fail(f"неизвестный источник трека: {source}",
             "допустимые значения track.source: video, gpx, nmea")

    pts = read_points(raw_gpx)
    if not pts:
        fail(
            "трек не содержит точек с отметками времени",
            "запись трека не велась либо не было решения ГНСС; "
            "без координат геопривязка невозможна, но остальные этапы "
            "могут быть выполнены при georef.enabled: false",
        )
    shutil.copy2(raw_gpx, stage_dir / "track.gpx")

    # --- показатели ---
    t_start, t_end = pts[0]["t"], pts[-1]["t"]
    gaps = [(b["t"] - a["t"]).total_seconds() for a, b in zip(pts, pts[1:])]
    steps = [haversine(a, b) for a, b in zip(pts, pts[1:])]
    length = sum(steps)
    max_gap = max(gaps) if gaps else 0.0
    median_step = sorted(steps)[len(steps) // 2] if steps else 0.0
    span_lat = (max(p["lat"] for p in pts) - min(p["lat"] for p in pts)) * 110540
    span_lon = (max(p["lon"] for p in pts) - min(p["lon"] for p in pts)) * 111320 * \
        math.cos(math.radians(pts[0]["lat"]))

    out.rule()
    out.kv("точек", len(pts))
    out.kv("интервал трека", f"{t_start.isoformat(timespec='seconds')} … "
                             f"{t_end.isoformat(timespec='seconds')}")
    out.kv("протяжённость", f"{length:.1f} м")
    out.kv("габарит полигона", f"{span_lon:.1f} \u00d7 {span_lat:.1f} м")
    out.kv("макс. разрыв", f"{max_gap:.1f} с")

    # --- проверки ---
    frames = json.loads((session.dir("02_frames") / "frames.json").read_text("utf-8"))
    f_start = datetime.fromisoformat(frames[0]["t_utc"]).astimezone(timezone.utc)
    f_end = datetime.fromisoformat(frames[-1]["t_utc"]).astimezone(timezone.utc)
    if f_start < t_start or f_end > t_end:
        lag_head = (t_start - f_start).total_seconds()
        lag_tail = (f_end - t_end).total_seconds()
        msg = (f"трек не покрывает интервал съёмки "
               f"(недостаёт {max(lag_head, 0):.1f} с в начале и "
               f"{max(lag_tail, 0):.1f} с в конце)")
        if max(lag_head, lag_tail) > cfg["track"]["max_uncovered_s"]:
            fail(msg,
                 "вероятная причина — рассогласование часов камеры и приёмника; "
                 "уточните time.clock_offset_s по снимку индикатора точного времени")
        rep.warn(msg, out)

    if max_gap > cfg["track"]["max_gap_s"]:
        rep.warn(f"разрыв записи {max_gap:.1f} с — соответствующие кадры "
                 f"координат не получат", out)
    jumps = [i for i, s in enumerate(steps)
             if median_step > 0 and s > cfg["track"]["jump_factor"] * median_step]
    if jumps:
        rep.warn(f"{len(jumps)} скачков положения (потенциально грубые ошибки, "
                 f"отбраковываются устойчивым решателем на этапе геопривязки)", out)
    if length < cfg["track"]["min_length_m"]:
        rep.warn(f"протяжённость трека {length:.1f} м мала для обхода объекта", out)

    qc_paths = plot_track(session, pts)
    rep.qc_images = [str(p.relative_to(session.root)) for p in qc_paths]
    out.qc(qc_paths, require_confirm=True)

    rep.params = {k: cfg["track"][k] for k in
                  ("source", "frequency_hz", "resampling_method", "timezone_offset_h")
                  if k in cfg["track"]}
    rep.metrics = {
        "n_points": len(pts),
        "t_start_utc": t_start.isoformat(),
        "t_end_utc": t_end.isoformat(),
        "length_m": round(length, 2),
        "span_m": [round(span_lon, 2), round(span_lat, 2)],
        "max_gap_s": round(max_gap, 2),
        "median_step_m": round(median_step, 3),
        "n_jumps": len(jumps),
    }
    rep.duration_s = out.done()
    rep.write(session, stage_dir)
    return rep
