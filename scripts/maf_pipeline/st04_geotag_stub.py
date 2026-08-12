#!/usr/bin/env python3
"""
st04_geotag.py — запись геопространственных данных в EXIF извлечённых кадров.

Режимы:
  track   — штатный: сопоставление кадров с GPX-треком по временным меткам
            (реализуется вызовом gpscorrelate)
  anchor  — отладочный: во все кадры записывается единственная координата,
            извлечённая из метаданных исходного видеофайла

ВНИМАНИЕ. В режиме anchor все кадры получают одинаковые координаты. Множество
соответствий "поза камеры <-> геодезические координаты" вырождается в точку,
вследствие чего определение параметров преобразования подобия (поворот, масштаб)
становится невозможным. Режим предназначен исключительно для проверки
работоспособности пайплайна и требует соответствующего режима на этапе
геопривязки. Экспериментальные результаты в этом режиме получены быть не могут.

Зависимости: exiftool (обязательно), ffprobe (для запасного способа извлечения)
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def require_tool(name):
    """Проверяет наличие внешней утилиты в PATH."""
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"Утилита '{name}' не найдена в PATH. "
            f"Установите её и повторите запуск."
        )
    return path


def run(cmd, check=True):
    """Запускает внешнюю команду и возвращает stdout."""
    proc = subprocess.run(
        cmd, capture_output=True, text=True, errors="replace"
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Команда завершилась с кодом {proc.returncode}:\n"
            f"  {' '.join(map(str, cmd))}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def to_decimal(value):
    """Приводит координату к десятичным градусам.

    Принимает: число; строку с десятичной записью; строку EXIF-вида
    "55/1 45/1 21/1"; строку вида "55 deg 45' 21.00\" N".
    """
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()

    # Десятичная запись
    try:
        return float(s)
    except ValueError:
        pass

    # Знак направления в конце или начале строки
    sign = 1.0
    m = re.search(r"\b([NSEW])\b|([NSEW])\s*$", s, flags=re.IGNORECASE)
    if m:
        letter = (m.group(1) or m.group(2)).upper()
        if letter in ("S", "W"):
            sign = -1.0

    # Числа в виде рациональных дробей или десятичных значений
    nums = []
    for token in re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", s):
        if "/" in token:
            num, den = token.split("/")
            den = float(den)
            if den == 0:
                return None
            nums.append(float(num) / den)
        else:
            nums.append(float(token))

    if not nums:
        return None

    deg = nums[0]
    minutes = nums[1] if len(nums) > 1 else 0.0
    seconds = nums[2] if len(nums) > 2 else 0.0

    if deg < 0:
        sign = -1.0
        deg = abs(deg)

    return sign * (deg + minutes / 60.0 + seconds / 3600.0)


def parse_iso6709(text):
    """Разбирает строку ISO 6709.

    Поддерживает записи вида "+55.7558+037.6173+156.000/",
    "+55+037/", "+55.7558+037.6173/CRSWGS_84".
    """
    s = str(text).strip().rstrip("/")
    s = re.sub(r"CRS[\w_]*$", "", s).rstrip("/")

    nums = re.findall(r"[+-]\d+(?:\.\d+)?", s)
    if len(nums) < 2:
        return None

    return {
        "lat": float(nums[0]),
        "lon": float(nums[1]),
        "alt": float(nums[2]) if len(nums) > 2 else None,
    }


# ---------------------------------------------------------------------------
# Извлечение опорной координаты из видеофайла
# ---------------------------------------------------------------------------

ISO6709_KEYS = (
    "com.apple.quicktime.location.ISO6709",
    "location",
    "location-eng",
    "LocationInformation",
    "GPSCoordinates",
)


def anchor_from_exiftool(video):
    """Извлекает координату утилитой exiftool.

    Ключ -ee включает разбор встроенных телеметрических потоков, что
    существенно для камер, записывающих трек в состав видеофайла.
    """
    out = run([
        "exiftool", "-ee", "-json", "-n", "-a", "-G1", str(video)
    ], check=False)
    if not out.strip():
        return None

    try:
        records = json.loads(out)
    except json.JSONDecodeError:
        return None

    lat = lon = alt = None
    for record in records:
        for key, value in record.items():
            short = key.split(":")[-1]

            if short in ISO6709_KEYS and lat is None:
                parsed = parse_iso6709(value)
                if parsed:
                    lat, lon = parsed["lat"], parsed["lon"]
                    alt = parsed["alt"] if alt is None else alt
                continue

            if short == "GPSLatitude" and lat is None:
                lat = to_decimal(value)
            elif short == "GPSLongitude" and lon is None:
                lon = to_decimal(value)
            elif short == "GPSAltitude" and alt is None:
                alt = to_decimal(value)

        if lat is not None and lon is not None:
            break

    if lat is None or lon is None:
        return None
    return {"lat": lat, "lon": lon, "alt": alt, "source": "exiftool"}


def anchor_from_ffprobe(video):
    """Запасной способ: разбор тегов контейнера утилитой ffprobe.

    Читает только теги контейнера; встроенные телеметрические потоки
    этим способом недоступны.
    """
    out = run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video)
    ], check=False)
    if not out.strip():
        return None

    try:
        meta = json.loads(out)
    except json.JSONDecodeError:
        return None

    tags = dict(meta.get("format", {}).get("tags", {}))
    for stream in meta.get("streams", []):
        for key, value in stream.get("tags", {}).items():
            tags.setdefault(key, value)

    # Поиск без учёта регистра
    lowered = {k.lower(): v for k, v in tags.items()}

    for key in ISO6709_KEYS:
        value = lowered.get(key.lower())
        if value:
            parsed = parse_iso6709(value)
            if parsed:
                parsed["source"] = "ffprobe"
                return parsed

    lat = to_decimal(lowered["gpslatitude"]) if "gpslatitude" in lowered else None
    lon = to_decimal(lowered["gpslongitude"]) if "gpslongitude" in lowered else None
    if lat is None or lon is None:
        return None

    if str(lowered.get("gpslatituderef", "")).upper().startswith("S"):
        lat = -abs(lat)
    if str(lowered.get("gpslongituderef", "")).upper().startswith("W"):
        lon = -abs(lon)

    alt = to_decimal(lowered["gpsaltitude"]) if "gpsaltitude" in lowered else None
    return {"lat": lat, "lon": lon, "alt": alt, "source": "ffprobe"}


def dump_available_tags(video, limit=40):
    """Выводит доступные метаданные — для диагностики при неудаче."""
    out = run(["exiftool", "-ee", "-G1", "-a", "-s", str(video)], check=False)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    print("\nДоступные метаданные видеофайла:", file=sys.stderr)
    for line in lines[:limit]:
        print(f"  {line}", file=sys.stderr)
    if len(lines) > limit:
        print(f"  ... ещё {len(lines) - limit} полей", file=sys.stderr)


# ---------------------------------------------------------------------------
# Запись координат в кадры
# ---------------------------------------------------------------------------

def write_anchor(frames_dir, coord):
    """Записывает одну координату во все изображения каталога."""
    frames = sorted(
        p for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frames:
        raise RuntimeError(f"В каталоге {frames_dir} не найдено изображений.")

    lat, lon, alt = coord["lat"], coord["lon"], coord.get("alt")

    cmd = [
        "exiftool",
        "-overwrite_original",
        "-n",
        f"-GPSLatitude={abs(lat)}",
        f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
        f"-GPSLongitude={abs(lon)}",
        f"-GPSLongitudeRef={'E' if lon >= 0 else 'W'}",
    ]
    if alt is not None:
        cmd += [
            f"-GPSAltitude={abs(alt)}",
            f"-GPSAltitudeRef={0 if alt >= 0 else 1}",
        ]
    # Признак отладочного происхождения координат — остаётся в файле
    cmd += ["-XMP:Description=GEOTAG_MODE=anchor_stub"]
    cmd += [str(p) for p in frames]

    run(cmd)
    return frames


def write_from_track(frames_dir, gpx, time_offset=0, max_gap=None,
                     no_interpolate=False):
    """Штатный режим: сопоставление кадров с треком через gpscorrelate."""
    require_tool("gpscorrelate")

    frames = sorted(
        p for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frames:
        raise RuntimeError(f"В каталоге {frames_dir} не найдено изображений.")

    cmd = ["gpscorrelate", "--gps", str(gpx)]
    if time_offset:
        cmd += ["--photooffset", str(int(time_offset))]
    if max_gap is not None:
        cmd += ["--max-dist", str(int(max_gap))]
    if no_interpolate:
        cmd += ["--no-interpolation"]
    cmd += [str(p) for p in frames]

    run(cmd)
    return frames


def count_tagged(frames):
    """Возвращает число файлов, в которых присутствуют координаты."""
    out = run(
        ["exiftool", "-n", "-json", "-GPSLatitude", "-GPSLongitude"]
        + [str(p) for p in frames],
        check=False,
    )
    if not out.strip():
        return 0
    try:
        records = json.loads(out)
    except json.JSONDecodeError:
        return 0
    return sum(
        1 for r in records
        if r.get("GPSLatitude") is not None and r.get("GPSLongitude") is not None
    )


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("frames", help="каталог с извлечёнными кадрами")
    ap.add_argument("--mode", choices=["track", "anchor"], default="track",
                    help="источник координат (по умолчанию track)")
    ap.add_argument("--gpx", help="файл трека GPX (режим track)")
    ap.add_argument("--video", help="исходный видеофайл (режим anchor)")
    ap.add_argument("--lat", type=float,
                    help="широта, град. — задаётся вручную вместо извлечения")
    ap.add_argument("--lon", type=float, help="долгота, град.")
    ap.add_argument("--alt", type=float, help="высота, м")
    ap.add_argument("--time-offset", type=int, default=0,
                    help="временная поправка камеры, с (режим track)")
    ap.add_argument("--max-gap", type=int,
                    help="макс. интервал интерполяции, с (режим track)")
    ap.add_argument("--no-interpolate", action="store_true",
                    help="брать ближайшую отметку трека без интерполяции")
    ap.add_argument("--report", help="путь для сохранения отчёта в JSON")
    args = ap.parse_args()

    require_tool("exiftool")
    frames_dir = Path(args.frames)
    if not frames_dir.is_dir():
        sys.exit(f"Каталог не найден: {frames_dir}")

    report = {
        "stage": "st04_geotag",
        "mode": args.mode,
        "frames_dir": str(frames_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if args.mode == "track":
        if not args.gpx:
            sys.exit("Режим track требует указания --gpx")
        frames = write_from_track(
            frames_dir, args.gpx, args.time_offset,
            args.max_gap, args.no_interpolate,
        )
        report.update({
            "gpx": str(args.gpx),
            "time_offset_s": args.time_offset,
        })

    else:  # anchor
        if args.lat is not None and args.lon is not None:
            coord = {"lat": args.lat, "lon": args.lon,
                     "alt": args.alt, "source": "manual"}
        else:
            if not args.video:
                sys.exit("Режим anchor требует --video либо пары --lat/--lon")
            require_tool("ffprobe")
            coord = anchor_from_exiftool(args.video)
            if coord is None:
                coord = anchor_from_ffprobe(args.video)
            if coord is None:
                print("Координаты в метаданных видеофайла не обнаружены.",
                      file=sys.stderr)
                dump_available_tags(args.video)
                sys.exit(
                    "\nЗадайте координату вручную ключами --lat и --lon."
                )

        frames = write_anchor(frames_dir, coord)
        report["anchor"] = coord
        report["warning"] = (
            "Все кадры содержат одинаковые координаты. Определение параметров "
            "преобразования подобия невозможно; этап геопривязки должен "
            "выполняться в режиме anchor. Режим непригоден для получения "
            "экспериментальных результатов."
        )

    tagged = count_tagged(frames)
    report.update({"frames_total": len(frames), "frames_tagged": tagged})

    print(f"Режим:            {args.mode}")
    if args.mode == "anchor":
        c = report["anchor"]
        alt = f", {c['alt']} м" if c.get("alt") is not None else ""
        print(f"Опорная точка:    {c['lat']:.6f}, {c['lon']:.6f}{alt}")
        print(f"Источник:         {c['source']}")
    print(f"Кадров обработано: {len(frames)}")
    print(f"Кадров с координатами: {tagged}")

    if tagged < len(frames):
        print(f"\nВНИМАНИЕ: у {len(frames) - tagged} кадров координаты "
              f"отсутствуют.", file=sys.stderr)
    if args.mode == "anchor":
        print("\nВНИМАНИЕ: отладочный режим. " + report["warning"],
              file=sys.stderr)

    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()