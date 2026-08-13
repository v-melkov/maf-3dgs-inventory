#!/usr/bin/env python3
"""
st00_init.py — создание структуры каталогов проекта обработки объекта учёта.

Формирует дерево каталогов по этапам пайплайна, манифест проекта, шаблон
описания линейного эталона и бланк контрольных измерений.

Пример:
    python st00_init.py bench --title "Скамейка типовая" \\
        --type bench --complexity low --markers 0 1

    python st00_init.py dino --title "Скульптура (динозавр)" \\
        --type sculpture --complexity high --markers 0 1 2
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# --- Структура проекта -----------------------------------------------------
# Порядок соответствует этапам пайплайна. При изменении состава этапов
# правится только этот список.

STAGES = [
    ("00_source", "исходное видео, трек ГНСС, полевые заметки"),
    ("01_reframe", "кадры в перспективной проекции (при панорамной съёмке)"),
    ("02_frames", "кадры, отобранные по критерию резкости; сюда же пишутся координаты EXIF"),
    ("03_colmap", "результаты оценки поз камер"),
    ("04_masks", "бинарные маски объекта учёта"),
    ("05_model", "обученная 3DGS-модель"),
    ("06_scale", "детектирование маркеров, масштабный коэффициент"),
    ("07_object", "гауссианы объекта: перенос масок, фильтрация, OBB"),
    ("08_georef", "параметры геопривязки, геопривязанная модель"),
    ("09_output", "инвентарная карточка, слой GeoPackage, экспорт для ГИС"),
    ("reports", "машиночитаемые отчёты по этапам (JSON)"),
    ("reference", "контрольные измерения, схемы, фотофиксация"),
    ("logs", "журналы выполнения"),
]

SUBDIRS = {
    "03_colmap": ["sparse", "undistorted"],
    "07_object": ["raw", "filtered"],
}

COMPLEXITY = {
    "low": "низкая",
    "medium": "средняя",
    "high": "высокая",
}

OBJECT_TYPES = [
    "bench", "bin", "lamp", "fence", "planter", "sculpture",
    "sign", "waste_site", "playground", "other",
]


def build_manifest(slug, args):
    """Формирует манифест проекта."""
    return {
        "project": slug,
        "title": args.title or slug,
        "object": {
            "type": args.type,
            "complexity": args.complexity,
            "complexity_ru": COMPLEXITY.get(args.complexity),
        },
        "survey": {
            "date": args.date,
            "operator": args.operator,
            "site": args.site,
            "camera": None,
            "gnss_source": None,
            "notes": None,
        },
        "etalon": {
            "type": "нивелирная рейка",
            "marker_dictionary": "DICT_5X5",
            "marker_size_mm": 100,
            "markers": [
                {"id": i, "nominal_position_m": pos}
                for i, pos in enumerate(args.markers)
            ],
            "measured_distances_mm": None,
            "comment": (
                "measured_distances_mm заполняется по результатам прямого "
                "измерения расстояний между центрами маркеров после их "
                "закрепления. Формат: {\"0-1\": 998.5, \"0-2\": 1997.0}. "
                "Номинальные значения для расчёта не используются."
            ),
        },
        "pipeline": {
            "geotag_mode": None,
            "scale_factor": None,
            "scale_rmse": None,
            "georef_rmse": None,
            "stages_completed": [],
        },
        "created": date.today().isoformat(),
    }


ETALON_README = """\
# Линейный эталон

## Что заполнить перед обработкой

1. Закрепить маркеры на эталоне согласно схеме проекта.
2. Измерить рулеткой фактические расстояния **между центрами маркеров**
   (не по делениям шкалы рейки — маркеры смещены относительно её оси).
3. Внести измеренные значения в `project.json`, раздел `etalon`,
   поле `measured_distances_mm`.

Пример:

    "measured_distances_mm": {
      "0-1": 998.5,
      "0-2": 1997.0,
      "1-2": 998.5
    }

## Контроль

- Телескопическая рейка фиксируется в рабочем положении на всю сессию.
- Взаимное положение маркеров и рейки не должно меняться между съёмками.
- При съёмке нескольких объектов одной сессией эталон размещается
  в 2–3 различных положениях для контроля постоянства масштаба.
"""

REFERENCE_CSV = """\
parameter,nominal_mm,measured_mm,instrument,operator,date,notes
length,,,рулетка,,,
width,,,рулетка,,,
height,,,рулетка,,,
"""

PROJECT_README = """\
# Проект: {title}

Объект: {type} ({complexity})
Создан: {created}

## Порядок работы

Каждый этап пишет результат в свой каталог и отчёт в `reports/`.
Параметры и итоговые показатели накапливаются в `project.json`.

{stages}

## Перед началом обработки

- [ ] Исходное видео и трек помещены в `00_source/`
- [ ] Контрольные измерения объекта внесены в `reference/measurements.csv`
- [ ] Фактические расстояния между маркерами внесены в `project.json`
- [ ] В `project.json` заполнены сведения о съёмке (камера, источник ГНСС)
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("slug", help="краткое имя проекта (латиницей, без пробелов)")
    ap.add_argument("--root", default="projects",
                    help="корневой каталог проектов (по умолчанию projects)")
    ap.add_argument("--title", help="наименование объекта")
    ap.add_argument("--type", choices=OBJECT_TYPES, default="other",
                    help="тип объекта учёта")
    ap.add_argument("--complexity", choices=list(COMPLEXITY), default="low",
                    help="уровень геометрической сложности")
    ap.add_argument("--markers", type=float, nargs="+", default=[0, 1],
                    help="номинальные позиции маркеров на эталоне, м")
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="дата съёмки (ГГГГ-ММ-ДД)")
    ap.add_argument("--operator", help="исполнитель обследования")
    ap.add_argument("--site", help="наименование съёмочного полигона")
    ap.add_argument("--force", action="store_true",
                    help="дописать структуру в существующий каталог")
    ap.add_argument("--dry-run", action="store_true",
                    help="показать план без создания файлов")
    args = ap.parse_args()

    root = Path(args.root) / args.slug

    if root.exists() and not args.force and not args.dry_run:
        sys.exit(
            f"Каталог {root} уже существует. "
            f"Используйте --force для дополнения структуры."
        )

    manifest = build_manifest(args.slug, args)

    stage_lines = "\n".join(
        f"- `{name}/` — {desc}" for name, desc in STAGES
    )
    readme = PROJECT_README.format(
        title=manifest["title"],
        type=args.type,
        complexity=COMPLEXITY[args.complexity],
        created=manifest["created"],
        stages=stage_lines,
    )

    # Состав операций
    dirs = [root]
    for name, _ in STAGES:
        dirs.append(root / name)
        for sub in SUBDIRS.get(name, []):
            dirs.append(root / name / sub)

    files = {
        root / "project.json": json.dumps(manifest, ensure_ascii=False, indent=2)
                               + "\n",
        root / "README.md": readme,
        root / "reference" / "measurements.csv": REFERENCE_CSV,
        root / "reference" / "etalon.md": ETALON_README,
    }

    if args.dry_run:
        print("Будут созданы каталоги:")
        for d in dirs:
            print(f"  {d}/")
        print("\nБудут созданы файлы:")
        for f in files:
            print(f"  {f}")
        return

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    created, skipped = [], []
    for path, content in files.items():
        if path.exists():
            skipped.append(path)
            continue
        path.write_text(content, encoding="utf-8")
        created.append(path)

    # Пустые каталоги сохраняются в системе контроля версий
    for d in dirs:
        if d != root and not any(d.iterdir()):
            (d / ".gitkeep").touch()

    print(f"Проект создан: {root}")
    print(f"Каталогов:     {len(dirs)}")
    print(f"Файлов:        {len(created)}"
          + (f" (пропущено существующих: {len(skipped)})" if skipped else ""))
    print("\nДалее:")
    print(f"  1. Поместите видео в {root / '00_source'}/")
    print(f"  2. Заполните контрольные измерения "
          f"в {root / 'reference'vj / 'measurements.csv'}")
    print(f"  3. Внесите расстояния между маркерами в {root / 'project.json'}")


if __name__ == "__main__":
    main()