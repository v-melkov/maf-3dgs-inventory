"""Этап 04 — присвоение координат кадрам.

Сопоставление кадров с треком выполняется по временным меткам с линейной
интерполяцией между отметками. Реализация — exiftool в режиме -geotag;
процедура полностью соответствует описанной в п. 2.4.4.5 методики.

Время кадров записано в UTC с явным нулевым смещением (этап 02), поэтому
преобразования часовых поясов здесь не выполняются и служить источником
ошибок не могут.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from .core import Out, Session, StageReport, fail, require_tool, run, tool_version

STAGE = "04_geotag"


def _read_gps_table(images_dir: Path, log: Path) -> list[dict]:
    """Машиночитаемая выгрузка присвоенных координат.

    Возвращаемая таблица используется и для контроля, и на этапе
    геопривязки — повторное чтение EXIF там не требуется.
    """
    proc = run(
        ["exiftool", "-n", "-csv",
         "-FileName", "-DateTimeOriginal", "-SubSecTimeOriginal",
         "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", "-GPSDOP",
         str(images_dir)],
        log_path=log, ok_codes=(0, 1),
    )
    rows = list(csv.DictReader(io.StringIO(proc.stdout)))
    table = []
    for r in rows:
        def num(key):
            v = r.get(key, "")
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        table.append({
            "name": r.get("FileName"),
            "lat": num("GPSLatitude"),
            "lon": num("GPSLongitude"),
            "alt": num("GPSAltitude"),
            "dop": num("GPSDOP"),
        })
    return table


def run_stage(session: Session, cfg: dict, force: bool = False) -> StageReport:
    session.require_stage("03_track")
    rep = StageReport(STAGE)
    out = Out(STAGE, "Присвоение координат кадрам", "exiftool")
    log = session.log_path(STAGE)
    stage_dir = session.dir(STAGE)

    require_tool("exiftool", "загрузите exiftool с exiftool.org и добавьте в PATH")
    rep.tool_versions["exiftool"] = tool_version(["exiftool", "-ver"])

    images = session.dir("02_frames/selected")
    gpx = session.dir("03_track") / "track.gpx"
    max_int = int(cfg["geotag"]["max_interpolate_s"])
    max_ext = int(cfg["geotag"]["max_extrapolate_s"])

    cmd = [
        "exiftool",
        "-geotag", str(gpx),
        "-geotime<${DateTimeOriginal}${OffsetTimeOriginal}",
        "-api", f"GeoMaxIntSecs={max_int}",
        "-api", f"GeoMaxExtSecs={max_ext}",
        "-overwrite_original",
        "-preserve",
        str(images),
    ]
    # Код 1 возвращается, когда часть файлов не удалось обработать; это не
    # отказ, а основание для анализа доли непокрытых кадров ниже.
    run(cmd, log_path=log, ok_codes=(0, 1, 2), echo=True)

    table = _read_gps_table(images, log)
    (stage_dir / "image_coordinates.csv").write_text(
        "name,lat,lon,alt,dop\n" + "\n".join(
            f"{r['name']},{r['lat']},{r['lon']},{r['alt']},{r['dop']}" for r in table
        ),
        "utf-8",
    )

    total = len(table)
    tagged = [r for r in table if r["lat"] is not None and r["lon"] is not None]
    share = len(tagged) / total if total else 0.0

    out.kv("кадров", total)
    out.kv("получили координаты", f"{len(tagged)} ({100 * share:.1f} %)")

    if total == 0:
        fail("в рабочем наборе нет кадров", "выполните этап 02")
    if share < cfg["geotag"]["min_share_fail"]:
        fail(
            f"координаты присвоены лишь {100 * share:.1f} % кадров",
            "наиболее вероятная причина — рассогласование шкал времени; "
            "проверьте time.clock_offset_s и интервал трека в отчёте этапа 03",
        )
    if share < cfg["geotag"]["min_share_warn"]:
        rep.warn(f"без координат остались {total - len(tagged)} кадров", out)

    # --- разброс: защита от вырожденных результатов ---
    if tagged:
        lat0 = sum(r["lat"] for r in tagged) / len(tagged)
        kx = 111320 * math.cos(math.radians(lat0))
        span_lon = (max(r["lon"] for r in tagged) - min(r["lon"] for r in tagged)) * kx
        span_lat = (max(r["lat"] for r in tagged) - min(r["lat"] for r in tagged)) * 110540
        span = max(span_lon, span_lat)
        out.kv("разброс точек съёмки", f"{span_lon:.1f} \u00d7 {span_lat:.1f} м")
        if span < cfg["geotag"]["min_span_m"]:
            rep.warn(
                f"разброс точек съёмки {span:.1f} м близок к нулю — вероятно, "
                "всем кадрам присвоена одна отметка трека", out,
            )
        if span > cfg["geotag"]["max_span_m"]:
            rep.warn(
                f"разброс точек съёмки {span:.0f} м несоразмерен обходу одного "
                "объекта — проверьте сопоставление по времени", out,
            )
        track_span = session.report_of("03_track")["metrics"]["span_m"]
        ratio = span / max(max(track_span), 1e-6)
        if ratio > 1.5:
            rep.warn(
                f"разброс точек съёмки превышает габарит трека в {ratio:.1f} раза", out
            )

    rep.params = {"max_interpolate_s": max_int, "max_extrapolate_s": max_ext}
    rep.metrics = {
        "n_images": total,
        "n_tagged": len(tagged),
        "share_tagged": round(share, 4),
        "span_m": [round(span_lon, 2), round(span_lat, 2)] if tagged else None,
    }
    rep.duration_s = out.done()
    rep.write(session, stage_dir)
    return rep
