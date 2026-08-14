"""Общие механизмы пайплайна: манифест сессии, запуск внешних программ,
отчёты этапов и форматированный вывод.

Все этапы импортируют отсюда и не обращаются друг к другу напрямую:
связь между этапами — только через артефакты на диске и session.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Исключения
# ---------------------------------------------------------------------------


class StageError(RuntimeError):
    """Отказ этапа. Сообщение адресовано оператору и должно содержать
    что проверялось, что получено и что требуется сделать."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class StageSkipped(Exception):
    """Этап уже выполнен и повтор не запрошен."""


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------

_RULE = "\u2500" * 64


class Out:
    """Формирование вывода этапа: числа, а не сообщения о ходе работы."""

    def __init__(self, stage: str, title: str, tool: str | None = None):
        self.stage = stage
        head = f"[{stage}] {title}"
        if tool:
            head += f" ({tool})"
        print()
        print(head)
        self._t0 = time.time()

    def kv(self, key: str, value: Any) -> None:
        print(f"  {key + ':':<26}{value}")

    def rule(self) -> None:
        print("  " + _RULE)

    def step(self, name: str, value: Any, seconds: float | None = None) -> None:
        t = f"   {fmt_hms(seconds)}" if seconds is not None else ""
        print(f"  {name:<30}{str(value):<14}{t}")

    def warn(self, message: str) -> None:
        print(f"  \u26a0 {message}")

    def qc(self, paths: Sequence[Path], require_confirm: bool = False) -> None:
        if not paths:
            return
        rel = ", ".join(str(p) for p in paths)
        print(f"  контроль: {rel}")
        if require_confirm:
            print("  \u2192 просмотрите материалы контроля перед продолжением")

    def done(self) -> float:
        dt = time.time() - self._t0
        print(f"  выполнено за {fmt_hms(dt)}")
        return dt


def fmt_hms(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def fail(message: str, hint: str | None = None) -> None:
    raise StageError(message, hint)


# ---------------------------------------------------------------------------
# Файлы
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path) -> None:
    """Жёсткая ссылка, если возможно (экономит место и время на тысячах
    кадров), иначе копирование. Символические ссылки под Windows требуют
    прав администратора, поэтому не используются."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def ensure_free_space(path: Path, need_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < need_bytes:
        fail(
            f"недостаточно места: требуется {need_bytes / 2**30:.1f} ГиБ, "
            f"доступно {free / 2**30:.1f} ГиБ",
            "освободите место или укажите другой каталог данных",
        )


def utcnow() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Запуск внешних программ
# ---------------------------------------------------------------------------


def which(name: str) -> str | None:
    return shutil.which(name)


def require_tool(name: str, hint: str) -> str:
    p = shutil.which(name)
    if not p:
        fail(f"не найдена программа '{name}'", hint)
    return p


def run(
    cmd: Sequence[str],
    log_path: Path | None = None,
    cwd: Path | None = None,
    check: bool = True,
    ok_codes: Iterable[int] = (0,),
    echo: bool = False,
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess:
    """Запуск внешней программы с записью полного вывода в журнал.

    В журнал всегда пишется сама команда — это делает журнал сессии
    воспроизводимым протоколом обработки, пригодным для приложения к работе.
    """
    cmd = [str(c) for c in cmd]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n$ {subprocess.list2cmdline(cmd)}\n")
    if echo:
        print("  $ " + subprocess.list2cmdline(cmd))

    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            if proc.stdout:
                f.write(proc.stdout)
            if proc.stderr:
                f.write("\n--- stderr ---\n" + proc.stderr)
    if check and proc.returncode not in tuple(ok_codes):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        fail(
            f"'{cmd[0]}' завершилась с кодом {proc.returncode}\n      "
            + "\n      ".join(tail),
            f"полный вывод: {log_path}" if log_path else None,
        )
    return proc


def tool_version(cmd: Sequence[str]) -> str:
    try:
        p = subprocess.run(
            [str(c) for c in cmd], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60,
        )
        text = (p.stdout or p.stderr or "").strip()
        return text.splitlines()[0][:120] if text else "?"
    except Exception as exc:  # noqa: BLE001
        return f"недоступна ({exc.__class__.__name__})"


# ---------------------------------------------------------------------------
# Манифест сессии
# ---------------------------------------------------------------------------

STAGES = [
    ("01_raw", "Приём исходного материала"),
    ("02_frames", "Извлечение и отбор кадров"),
    ("03_track", "Подготовка трека ГНСС"),
    ("04_geotag", "Присвоение координат кадрам"),
    ("05_sfm", "Оценка поз камер"),
    ("06_objects", "Сегментация объектов"),
    ("07_models", "Обучение 3DGS-моделей"),
    ("08_scale",   "Масштабный коэффициент по эталону"),
    ("09_object",  "Гауссианы объекта: маски в 3D, фильтрация, OBB"),
    ("10_georef",  "Геопривязка модели"),
    ("11_card",    "Инвентарная карточка"),
    ("12_export",  "Экспорт в ГИС"),
]


class Session:
    """Каталог сессии = один обход = один объект учёта плюс эталон."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "session.json"
        self.data: dict[str, Any] = {}
        if self.manifest_path.exists():
            self.data = json.loads(self.manifest_path.read_text("utf-8"))

    # --- пути ---
    @property
    def id(self) -> str:
        return self.data.get("session_id", self.root.name)

    def dir(self, name: str, create: bool = True) -> Path:
        p = self.root / name
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def log_path(self, stage: str) -> Path:
        d = self.dir("logs")
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        return d / f"{stamp}_{stage}.log"

    def object_dir(self, object_id: str, create: bool = True) -> Path:
        p = self.root / "06_objects" / object_id
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def object_stage_dir(self, stage: str, object_id: str, create: bool = True) -> Path:
        p = self.root / stage / object_id
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    # --- манифест ---
    def init(self, session_id: str, config: dict[str, Any]) -> None:
        self.data.setdefault("session_id", session_id)
        self.data.setdefault("created", utcnow())
        self.data["config"] = config
        self.data.setdefault("stages", {})
        self.save()

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), "utf-8"
        )

    def status(self, stage: str) -> str:
        return self.data.get("stages", {}).get(stage, {}).get("status", "не выполнен")

    def require_stage(self, stage: str) -> None:
        st = self.status(stage)
        if st not in ("ok", "ok_with_warnings"):
            fail(
                f"этап {stage} не выполнен (состояние: {st})",
                f"выполните: python run.py {stage.split('_')[0]} {self.root}",
            )

    def record(self, report: "StageReport") -> None:
        self.data.setdefault("stages", {})[report.stage] = {
            "status": report.status,
            "finished": utcnow(),
            "duration_s": round(report.duration_s, 1),
            "metrics": report.metrics,
            "warnings": report.warnings,
        }
        self.save()

    def report_of(self, stage: str) -> dict[str, Any]:
        p = self.root / stage / f"{stage}_report.json"
        if not p.exists():
            fail(f"отсутствует отчёт этапа {stage}: {p}")
        return json.loads(p.read_text("utf-8"))


class StageReport:
    def __init__(self, stage: str):
        self.stage = stage
        self.started = utcnow()
        self.status = "ok"
        self.duration_s = 0.0
        self.tool_versions: dict[str, str] = {}
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.qc_images: list[str] = []
        self.inputs_sha256: dict[str, str] = {}

    def warn(self, message: str, out: Out | None = None) -> None:
        self.warnings.append(message)
        self.status = "ok_with_warnings"
        if out:
            out.warn(message)

    def write(self, session: Session, stage_dir: Path) -> Path:
        stage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": self.stage,
            "status": self.status,
            "started": self.started,
            "duration_s": round(self.duration_s, 1),
            "tool_versions": self.tool_versions,
            "params": self.params,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "qc_images": self.qc_images,
            "inputs_sha256": self.inputs_sha256,
        }
        p = stage_dir / f"{self.stage}_report.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        session.record(self)
        return p


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


def load_config(defaults_path: Path, session_config: Path | None) -> dict[str, Any]:
    import yaml  # PyYAML

    cfg = yaml.safe_load(defaults_path.read_text("utf-8")) or {}
    if session_config and Path(session_config).exists():
        over = yaml.safe_load(Path(session_config).read_text("utf-8")) or {}
        cfg = deep_merge(cfg, over)
    return cfg


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
