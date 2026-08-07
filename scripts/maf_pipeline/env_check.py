"""Проверка окружения.

Выполняется до начала обработки и при развёртывании на новой машине.
Вывод пригоден для включения в приложение к работе как протокол
воспроизводимости: фиксирует версии всех компонентов обработки.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

from .core import Out, tool_version, utcnow

# Минимальная вычислительная способность для сборок PyTorch под CUDA 12.8.
# RTX 5070 — архитектура Blackwell, sm_120: колёса cu121/cu124 не содержат
# кода под эту архитектуру и обратного PTX-совместимого пути не имеют.
BLACKWELL = (12, 0)


def _check_python_package(name: str, import_name: str | None = None) -> tuple[bool, str]:
    mod = import_name or name
    try:
        m = __import__(mod)
        return True, getattr(m, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        return False, exc.__class__.__name__


def _check_torch() -> dict:
    info: dict = {"installed": False}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{exc.__class__.__name__}: {exc}"
        return info

    info["installed"] = True
    info["version"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["device"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["capability"] = f"sm_{cap[0]}{cap[1]}"
        info["capability_tuple"] = list(cap)
        try:
            arch_list = torch.cuda.get_arch_list()
        except Exception:  # noqa: BLE001
            arch_list = []
        info["arch_list"] = arch_list
        info["arch_supported"] = (
            not arch_list or f"sm_{cap[0]}{cap[1]}" in arch_list
        )
        # Пробное вычисление: наличие устройства ещё не означает, что
        # в сборке есть ядра под его архитектуру.
        try:
            a = torch.randn(256, 256, device="cuda")
            info["matmul_ok"] = bool(torch.isfinite((a @ a).sum()).item())
        except Exception as exc:  # noqa: BLE001
            info["matmul_ok"] = False
            info["matmul_error"] = f"{exc.__class__.__name__}: {exc}"
    return info


def _check_ffmpeg_filters() -> dict:
    res = {"ffmpeg": bool(shutil.which("ffmpeg")), "ffprobe": bool(shutil.which("ffprobe"))}
    if res["ffmpeg"]:
        try:
            p = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60,
            )
            res["zscale"] = " zscale " in (p.stdout or "")
        except Exception:  # noqa: BLE001
            res["zscale"] = False
    return res


def run_env_check(report_path: Path | None = None) -> dict:
    out = Out("env", "Проверка окружения")
    problems: list[str] = []
    warnings: list[str] = []

    info: dict = {
        "checked": utcnow(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
    }
    out.kv("система", info["platform"])
    out.kv("python", info["python"])
    out.rule()

    # --- внешние программы ---
    ff = _check_ffmpeg_filters()
    info["ffmpeg"] = ff
    out.kv("ffmpeg", tool_version(["ffmpeg", "-version"]) if ff["ffmpeg"] else "НЕ НАЙДЕНА")
    if not ff["ffmpeg"] or not ff["ffprobe"]:
        problems.append("ffmpeg/ffprobe не найдены в PATH — извлечение кадров невозможно")

    exiftool = shutil.which("exiftool")
    info["exiftool"] = tool_version(["exiftool", "-ver"]) if exiftool else None
    out.kv("exiftool", info["exiftool"] or "НЕ НАЙДЕНА")
    if not exiftool:
        problems.append(
            "exiftool не найдена — запись координат в EXIF невозможна "
            "(exiftool.org, распакуйте exiftool(-k).exe как exiftool.exe в PATH)"
        )

    colmap = shutil.which("colmap")
    info["colmap"] = tool_version(["colmap", "-h"]) if colmap else None
    out.kv("colmap", (info["colmap"] or "НЕ НАЙДЕНА")[:60])
    if not colmap:
        problems.append("colmap не найдена — оценка поз камер невозможна")

    lfs = shutil.which("lichtfeld-studio") or shutil.which("LichtFeld-Studio")
    info["lichtfeld"] = tool_version([lfs, "--version"]) if lfs else None
    out.kv("lichtfeld-studio", (info["lichtfeld"] or "НЕ НАЙДЕНА")[:60])
    if not lfs:
        warnings.append(
            "lichtfeld-studio не найдена в PATH — укажите путь в конфигурации (train.binary)"
        )

    # --- пакеты python ---
    out.rule()
    for pkg, imp in [
        ("sharp-frames", "sharp_frames"),
        ("PyYAML", "yaml"),
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("matplotlib", "matplotlib"),
        ("gpxpy", "gpxpy"),
    ]:
        ok, ver = _check_python_package(pkg, imp)
        info.setdefault("packages", {})[pkg] = ver if ok else None
        out.kv(pkg, ver if ok else "НЕ УСТАНОВЛЕН")
        if not ok:
            problems.append(f"пакет {pkg} не установлен")

    # --- GPU и модели ---
    out.rule()
    torch_info = _check_torch()
    info["torch"] = torch_info
    if not torch_info["installed"]:
        problems.append("PyTorch не установлен — сегментация невозможна")
        out.kv("torch", "НЕ УСТАНОВЛЕН")
    else:
        out.kv("torch", f"{torch_info['version']} (сборка CUDA {torch_info['cuda_build']})")
        if not torch_info["cuda_available"]:
            problems.append("CUDA недоступна для PyTorch")
            out.kv("CUDA", "НЕДОСТУПНА")
        else:
            out.kv("устройство", torch_info["device"])
            out.kv("вычисл. способность", torch_info["capability"])
            out.kv("архитектуры в сборке", ", ".join(torch_info.get("arch_list", [])) or "?")
            if not torch_info.get("arch_supported", True):
                problems.append(
                    f"сборка PyTorch не содержит кода под {torch_info['capability']}: "
                    "для Blackwell (sm_120) требуются колёса под CUDA 12.8 и выше "
                    "(--index-url https://download.pytorch.org/whl/cu128)"
                )
            if not torch_info.get("matmul_ok", False):
                problems.append(
                    "пробное вычисление на GPU не выполнено: "
                    + str(torch_info.get("matmul_error", "причина не определена"))
                )

    for pkg, imp in [("sam2", "sam2"), ("transformers", "transformers")]:
        ok, ver = _check_python_package(pkg, imp)
        info.setdefault("packages", {})[pkg] = ver if ok else None
        out.kv(pkg, ver if ok else "НЕ УСТАНОВЛЕН")
        if not ok:
            problems.append(f"пакет {pkg} не установлен — сегментация невозможна")

    ok, ver = _check_python_package("pyosmogps", "pyosmogps")
    info.setdefault("packages", {})["pyosmogps"] = ver if ok else None
    if not ok:
        warnings.append(
            "pyosmogps не установлен — извлечение трека из видеофайла недоступно; "
            "источником трека может служить готовый GPX"
        )

    # --- итог ---
    out.rule()
    info["problems"] = problems
    info["warnings"] = warnings
    for w in warnings:
        out.warn(w)
    if problems:
        print(f"  \u2717 не выполнено условий: {len(problems)}")
        for p in problems:
            print(f"      - {p}")
    else:
        print("  \u2713 окружение готово")

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(
            json.dumps(info, ensure_ascii=False, indent=2), "utf-8"
        )
        out.kv("протокол", report_path)
    out.done()
    return info
