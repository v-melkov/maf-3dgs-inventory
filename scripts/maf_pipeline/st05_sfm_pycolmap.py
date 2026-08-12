#!/usr/bin/env python3
"""
st05_sfm_pycolmap.py — оценка поз камер средствами pycolmap.

Выполняет извлечение и сопоставление характерных точек, инкрементальную
реконструкцию и выгрузку модели в текстовом формате, пригодном для
последующих этапов (в частности, для st06_scale.py).

Реализация не зависит от наличия исполняемого файла COLMAP в системе.
Параметры задаются объектами настроек с проверкой наличия полей, что
обеспечивает совместимость с различными версиями pycolmap.

Пример:
    python st05_sfm_pycolmap.py projects/bench/02_frames projects/bench/03_colmap \\
        --max-image-size 1600 --matcher sequential \\
        --report projects/bench/reports/st05.json
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pycolmap

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Совместимость с версиями интерфейса
# ---------------------------------------------------------------------------

def set_nested(obj, path, value):
    """Присваивает значение вложенному полю, если оно существует.

    Возвращает True при успешном присваивании.
    """
    target = obj
    parts = path.split(".")
    for name in parts[:-1]:
        if not hasattr(target, name):
            return False
        target = getattr(target, name)
    if not hasattr(target, parts[-1]):
        return False
    setattr(target, parts[-1], value)
    return True


def build_reader_options(camera_model):
    """Параметры чтения изображений: модель камеры."""
    if not hasattr(pycolmap, "ImageReaderOptions"):
        return None
    opts = pycolmap.ImageReaderOptions()
    if not set_nested(opts, "camera_model", camera_model):
        print(f"Поле camera_model недоступно; используется модель "
              f"по умолчанию.", file=sys.stderr)
    return opts


def build_extraction_options(max_image_size):
    """Параметры извлечения признаков: наибольший размер изображения."""
    if not hasattr(pycolmap, "FeatureExtractionOptions"):
        return None
    opts = pycolmap.FeatureExtractionOptions()
    for path in ("sift.max_image_size", "max_image_size"):
        if set_nested(opts, path, max_image_size):
            return opts
    print("Поле max_image_size недоступно; используется значение "
          "по умолчанию.", file=sys.stderr)
    return opts


def resolve_device(name):
    """Возвращает значение перечисления Device по имени."""
    if not hasattr(pycolmap, "Device"):
        return None
    return {
        "auto": getattr(pycolmap.Device, "auto", None),
        "cpu": getattr(pycolmap.Device, "cpu", None),
        "cuda": getattr(pycolmap.Device, "cuda", None),
    }.get(name)


def extract(db_path, image_dir, max_image_size, camera_model, device):
    """Извлечение характерных точек."""
    kwargs = {}

    reader = build_reader_options(camera_model)
    if reader is not None:
        kwargs["reader_options"] = reader

    extraction = build_extraction_options(max_image_size)
    if extraction is not None:
        kwargs["extraction_options"] = extraction

    if hasattr(pycolmap, "CameraMode"):
        kwargs["camera_mode"] = pycolmap.CameraMode.SINGLE

    dev = resolve_device(device)
    if dev is not None:
        kwargs["device"] = dev

    try:
        pycolmap.extract_features(db_path, image_dir, **kwargs)
    except TypeError as exc:
        # Версия интерфейса не принимает часть аргументов — повтор без них
        print(f"Часть параметров не поддерживается ({exc}); "
              f"повтор с параметрами по умолчанию.", file=sys.stderr)
        pycolmap.extract_features(db_path, image_dir)


def match(db_path, kind, overlap, device):
    """Сопоставление характерных точек.

    Для кадров, извлечённых из видеопотока, последовательное сопоставление
    предпочтительнее полного: оно учитывает порядок кадров и выполняется
    существенно быстрее при сопоставимом качестве.
    """
    dev = resolve_device(device)

    if kind == "sequential" and hasattr(pycolmap, "match_sequential"):
        kwargs = {}

        # Имя класса настроек различается между версиями
        for cls_name in ("SequentialMatchingOptions",
                         "SequentialPairingOptions",
                         "SequentialPairGeneratorOptions"):
            cls = getattr(pycolmap, cls_name, None)
            if cls is None:
                continue
            opts = cls()
            set_nested(opts, "overlap", overlap)
            set_nested(opts, "loop_detection", True)
            for key in ("matching_options", "pairing_options", "options"):
                kwargs = {key: opts}
                try:
                    if dev is not None:
                        pycolmap.match_sequential(db_path, device=dev, **kwargs)
                    else:
                        pycolmap.match_sequential(db_path, **kwargs)
                    return "sequential"
                except TypeError:
                    continue
            break

        try:
            if dev is not None:
                pycolmap.match_sequential(db_path, device=dev)
            else:
                pycolmap.match_sequential(db_path)
            return "sequential"
        except Exception as exc:
            print(f"Последовательное сопоставление недоступно ({exc}); "
                  f"выполняется полное.", file=sys.stderr)

    try:
        if dev is not None:
            pycolmap.match_exhaustive(db_path, device=dev)
        else:
            pycolmap.match_exhaustive(db_path)
    except TypeError:
        pycolmap.match_exhaustive(db_path)
    return "exhaustive"


def write_model(reconstruction, out_dir):
    """Сохраняет модель в двоичном и текстовом виде."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = out_dir.parent / (out_dir.name + "_txt")
    txt_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(reconstruction, "write_binary"):
        reconstruction.write_binary(str(out_dir))
    elif hasattr(reconstruction, "write"):
        reconstruction.write(str(out_dir))

    if hasattr(reconstruction, "write_text"):
        reconstruction.write_text(str(txt_dir))
    else:
        raise RuntimeError(
            "Версия pycolmap не поддерживает выгрузку модели в текстовом "
            "формате. Обновите пакет."
        )
    return txt_dir


def n_registered(reconstruction):
    """Число зарегистрированных изображений (метод либо свойство)."""
    value = getattr(reconstruction, "num_reg_images", None)
    if callable(value):
        return value()
    if value is not None:
        return value
    return len(reconstruction.images)


def track_length(point):
    """Длина трека точки с учётом различий интерфейса."""
    track = getattr(point, "track", None)
    if track is None:
        return None
    length = getattr(track, "length", None)
    if callable(length):
        return length()
    elements = getattr(track, "elements", None)
    if elements is not None:
        return len(elements)
    return None


def summarize(reconstruction, n_images_total):
    """Собирает показатели качества реконструкции."""
    n_reg = n_registered(reconstruction)
    points = reconstruction.points3D

    lengths = [t for t in (track_length(p) for p in points.values())
               if t is not None]
    errors = [p.error for p in points.values()
              if getattr(p, "error", -1) >= 0]

    return {
        "images_total": n_images_total,
        "images_registered": n_reg,
        "registered_pct": round(100 * n_reg / n_images_total, 1)
                          if n_images_total else 0.0,
        "points3D": len(points),
        "mean_track_length": round(sum(lengths) / len(lengths), 2)
                             if lengths else None,
        "mean_reproj_error_px": round(sum(errors) / len(errors), 3)
                                if errors else None,
        "cameras": len(reconstruction.cameras),
    }


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("images", help="каталог с кадрами")
    ap.add_argument("output", help="каталог для результатов реконструкции")
    ap.add_argument("--max-image-size", type=int, default=1600,
                    help="наибольший размер изображения при извлечении "
                         "признаков, пикс. (по умолчанию 1600)")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL",
                    help="модель камеры (по умолчанию SIMPLE_RADIAL)")
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"],
                    default="sequential",
                    help="способ сопоставления (по умолчанию sequential)")
    ap.add_argument("--overlap", type=int, default=10,
                    help="глубина последовательного сопоставления")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="вычислительное устройство")
    ap.add_argument("--force", action="store_true",
                    help="удалить существующие результаты перед запуском")
    ap.add_argument("--report", help="путь для сохранения отчёта в JSON")
    args = ap.parse_args()

    image_dir = Path(args.images)
    out_dir = Path(args.output)

    if not image_dir.is_dir():
        sys.exit(f"Каталог с кадрами не найден: {image_dir}")

    n_images = sum(1 for p in image_dir.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if n_images == 0:
        sys.exit(f"В каталоге {image_dir} не найдено изображений.")

    db_path = out_dir / "database.db"
    sparse_dir = out_dir / "sparse"

    if args.force:
        if db_path.exists():
            db_path.unlink()
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
    elif db_path.exists():
        sys.exit(f"База данных {db_path} уже существует. "
                 f"Используйте --force для повторного запуска.")

    out_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    print(f"pycolmap {pycolmap.__version__}, "
          f"CUDA: {getattr(pycolmap, 'has_cuda', 'неизвестно')}")
    print(f"Кадров на входе: {n_images}\n")

    timings = {}

    t0 = time.time()
    print("Извлечение характерных точек...")
    extract(db_path, image_dir, args.max_image_size,
            args.camera_model, args.device)
    timings["extraction_s"] = round(time.time() - t0, 1)
    print(f"  выполнено за {timings['extraction_s']} с\n")

    t0 = time.time()
    print(f"Сопоставление ({args.matcher})...")
    matcher_used = match(db_path, args.matcher, args.overlap, args.device)
    timings["matching_s"] = round(time.time() - t0, 1)
    print(f"  выполнено за {timings['matching_s']} с\n")

    t0 = time.time()
    print("Инкрементальная реконструкция...")
    maps = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir)
    timings["mapping_s"] = round(time.time() - t0, 1)
    print(f"  выполнено за {timings['mapping_s']} с\n")

    if not maps:
        sys.exit(
            "Реконструкция не построена. Вероятные причины: недостаточное "
            "перекрытие между кадрами, малое число кадров, отсутствие "
            "текстуры на объекте съёмки."
        )

    best_id = max(maps, key=lambda k: n_registered(maps[k]))
    best = maps[best_id]

    if len(maps) > 1:
        print(f"ВНИМАНИЕ: получено {len(maps)} несвязанных моделей. "
              f"Выбрана модель {best_id}.", file=sys.stderr)
        print("Реконструкция распалась на фрагменты — вероятна нехватка "
              "перекрытия на отдельных участках траектории.\n",
              file=sys.stderr)

    txt_dir = write_model(best, sparse_dir / str(best_id))

    stats = summarize(best, n_images)
    report = {
        "stage": "st05_sfm",
        "pycolmap_version": pycolmap.__version__,
        "matcher": matcher_used,
        "max_image_size": args.max_image_size,
        "camera_model": args.camera_model,
        "n_submodels": len(maps),
        "selected_submodel": int(best_id),
        "timings": timings,
        "model_text_dir": str(txt_dir),
        **stats,
    }

    print("Результаты реконструкции:")
    print(f"  зарегистрировано кадров: {stats['images_registered']} "
          f"из {stats['images_total']} ({stats['registered_pct']} %)")
    print(f"  точек в разрежённом облаке: {stats['points3D']}")
    if stats["mean_track_length"] is not None:
        print(f"  средняя длина трека: {stats['mean_track_length']}")
    if stats["mean_reproj_error_px"] is not None:
        print(f"  средняя невязка репроекции: "
              f"{stats['mean_reproj_error_px']} px")
    print(f"\nМодель в текстовом формате: {txt_dir}")

    if stats["registered_pct"] < 90:
        print(f"\nВНИМАНИЕ: зарегистрировано менее 90 % кадров. "
              f"Качество исходных данных требует проверки.", file=sys.stderr)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Отчёт сохранён: {args.report}")


if __name__ == "__main__":
    main()