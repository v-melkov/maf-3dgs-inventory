#!/usr/bin/env python3
"""Пайплайн учёта МАФ: этапы 01-07 (от исходной записи до .ply).

Использование:

    python run.py check-env
    python run.py init   data\\bench-01 --config config\\bench-01.yaml
    python run.py ingest data\\bench-01
    python run.py frames data\\bench-01
    python run.py track  data\\bench-01
    python run.py geotag data\\bench-01
    python run.py sfm    data\\bench-01
    python run.py segment data\\bench-01 --object target --prompt "wooden bench"
    python run.py train   data\\bench-01 --object target
    python run.py all     data\\bench-01 --object target --prompt "wooden bench"

Каждый этап идемпотентен: повторный запуск без --force при готовом
результате не пересчитывает то, что уже сделано.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maf_pipeline.core import Session, StageError, load_config  # noqa: E402
from maf_pipeline import (  # noqa: E402
    env_check, st01_ingest, st02_frames, st03_track, st04_geotag,
    st05_sfm, st06_segment, st07_train,
)

HERE = Path(__file__).resolve().parent
DEFAULTS = HERE / "config" / "defaults.yaml"


def _session(args) -> tuple[Session, dict]:
    s = Session(Path(args.session))
    cfg_path = Path(args.config) if getattr(args, "config", None) else None
    if cfg_path is None:
        stored = s.data.get("config_path")
        cfg_path = Path(stored) if stored else None
    cfg = load_config(DEFAULTS, cfg_path)
    if not s.manifest_path.exists():
        s.init(s.root.name, cfg)
    if cfg_path:
        s.data["config_path"] = str(cfg_path)
    s.data["config"] = cfg
    s.save()
    return s, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-env", help="проверка окружения")
    p.add_argument("--report", help="путь для сохранения протокола")

    for name, help_text in [
        ("init", "создать каталог сессии"),
        ("ingest", "этап 01 — приём исходного материала"),
        ("frames", "этап 02 — извлечение и отбор кадров"),
        ("track", "этап 03 — подготовка трека ГНСС"),
        ("geotag", "этап 04 — присвоение координат кадрам"),
        ("sfm", "этап 05 — оценка поз камер"),
        ("segment", "этап 06 — сегментация объекта"),
        ("train", "этап 07 — обучение 3DGS-модели"),
        ("all", "этапы 01-07 подряд"),
        ("status", "состояние сессии"),
    ]:
        q = sub.add_parser(name, help=help_text)
        q.add_argument("session", help="каталог сессии, например data\\bench-01")
        q.add_argument("--config", help="файл конфигурации сессии (yaml)")
        q.add_argument("--force", action="store_true", help="пересчитать этап заново")
        q.add_argument("--yes", action="store_true",
                       help="не запрашивать подтверждений (пакетный режим)")
        if name in ("segment", "train", "all"):
            q.add_argument("--object", default="target", help="идентификатор объекта")
            q.add_argument("--prompt", help="текстовый запрос детектора")

    args = ap.parse_args()

    if args.cmd == "check-env":
        info = env_check.run_env_check(Path(args.report) if args.report else None)
        return 1 if info["problems"] else 0

    session, cfg = _session(args)

    if args.cmd == "init":
        session.dir("01_raw")
        print(f"\nсессия создана: {session.root}")
        print(f"поместите видеофайл обхода в {session.dir('01_raw')}")
        print("затем: python run.py ingest " + str(session.root))
        return 0

    if args.cmd == "status":
        print(f"\nсессия {session.id}")
        from maf_pipeline.core import STAGES
        for stage, title in STAGES:
            st = session.status(stage)
            mark = {"ok": "\u2713", "ok_with_warnings": "\u26a0"}.get(st, " ")
            print(f"  {mark} {stage:<12} {title:<34} {st}")
        return 0

    try:
        if args.cmd in ("ingest", "all"):
            st01_ingest.run_stage(session, cfg, args.force)
        if args.cmd in ("frames", "all"):
            st02_frames.run_stage(session, cfg, args.force)
        if args.cmd in ("track", "all"):
            st03_track.run_stage(session, cfg, args.force)
        if args.cmd in ("geotag", "all"):
            st04_geotag.run_stage(session, cfg, args.force)
        if args.cmd in ("sfm", "all"):
            st05_sfm.run_stage(session, cfg, args.force, assume_yes=args.yes)
        if args.cmd in ("segment", "all"):
            st06_segment.run_stage(session, cfg, args.object, args.prompt,
                                   args.force, assume_yes=args.yes)
        if args.cmd in ("train", "all"):
            st07_train.run_stage(session, cfg, args.object, args.force)
    except StageError as exc:
        print(f"\n  \u2717 ОТКАЗ: {exc}")
        if exc.hint:
            print(f"    {exc.hint}")
        return 2
    except KeyboardInterrupt:
        print("\n  прервано оператором")
        return 130
    except Exception:  # noqa: BLE001
        print("\n  \u2717 непредвиденная ошибка:")
        traceback.print_exc()
        return 3

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
