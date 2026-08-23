#!/usr/bin/env python
"""Kaggle environment probe — run this FIRST on a Kaggle GPU notebook.

Purpose: capture everything needed to know whether the Physical Data Compiler
pipeline can run here, before spending GPU quota on a real run.

Usage on Kaggle:
    !python scripts/kaggle_probe.py

Then copy the ENTIRE output back so the environment can be verified.
It prints no secret values — only whether a secret is retrievable.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

SEP = "=" * 72
VERDICT: dict[str, str] = {}


def head(title: str) -> None:
    print(f"\n{SEP}\n== {title}\n{SEP}")


def kv(key: str, value: object) -> None:
    print(f"  {key:<28} {value}")


def sh(cmd: str, limit: int = 40) -> str:
    """Run a shell command, never raise."""
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=120
        )
        text = (out.stdout or "") + (out.stderr or "")
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        return "\n".join(lines[:limit]) if lines else "(no output)"
    except Exception as exc:  # noqa: BLE001
        return f"(failed: {exc})"


def ver(module: str) -> str:
    """Import a module and report its version, never raise."""
    try:
        m = importlib.import_module(module)
        return str(getattr(m, "__version__", "(no __version__)"))
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({type(exc).__name__})"


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


# ───────────────────────────────────────────────────────────── 1. HOST
def probe_host() -> None:
    head("1. HOST")
    kv("platform", platform.platform())
    kv("python", sys.version.replace("\n", " "))
    kv("executable", sys.executable)
    kv("cpu_count", os.cpu_count())
    kv("cwd", os.getcwd())
    kv("Path.home()", Path.home())

    # RAM (Linux)
    try:
        meminfo = Path("/proc/meminfo").read_text()
        total = next(l for l in meminfo.splitlines() if l.startswith("MemTotal"))
        avail = next(l for l in meminfo.splitlines() if l.startswith("MemAvailable"))
        kv("MemTotal", total.split(":")[1].strip())
        kv("MemAvailable", avail.split(":")[1].strip())
    except Exception:  # noqa: BLE001
        kv("RAM", "(unavailable — not Linux?)")

    for path in ("/kaggle/working", "/kaggle/input", "/tmp", "/root"):
        if Path(path).exists():
            try:
                usage = shutil.disk_usage(path)
                kv(f"disk {path}", f"free {human(usage.free)} / {human(usage.total)}")
            except Exception:  # noqa: BLE001
                kv(f"disk {path}", "(stat failed)")
        else:
            kv(f"disk {path}", "DOES NOT EXIST")

    on_kaggle = Path("/kaggle").exists()
    kv("running on Kaggle", on_kaggle)
    VERDICT["on_kaggle"] = "YES" if on_kaggle else "NO"


# ───────────────────────────────────────────────────────────── 2. GPU
def probe_gpu() -> None:
    head("2. GPU / CUDA")
    print("--- nvidia-smi -L ---")
    print(sh("nvidia-smi -L"))
    print("\n--- nvidia-smi (memory/driver) ---")
    print(sh("nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version "
             "--format=csv"))
    print("\n--- nvcc --version ---")
    print(sh("nvcc --version"))
    print("\n--- CUDA_HOME / env ---")
    for var in ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"):
        kv(var, os.environ.get(var, "(unset)"))

    print("\n--- torch ---")
    try:
        import torch

        kv("torch.__version__", torch.__version__)
        kv("torch.version.cuda", torch.version.cuda)
        kv("cuda.is_available()", torch.cuda.is_available())
        kv("device_count", torch.cuda.device_count())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                kv(f"  gpu[{i}]", f"{props.name} | {human(props.total_memory)} | "
                                  f"sm_{props.major}{props.minor}")
            VERDICT["gpu"] = f"YES ({torch.cuda.get_device_name(0)})"
        else:
            VERDICT["gpu"] = "NO — torch cannot see a GPU"
        is_cpu_build = "+cpu" in torch.__version__
        kv("is CPU-only build", is_cpu_build)
        if is_cpu_build:
            VERDICT["gpu"] = "NO — torch is a +cpu build, reinstall CUDA torch"
    except Exception as exc:  # noqa: BLE001
        kv("torch", f"MISSING ({exc})")
        VERDICT["gpu"] = "UNKNOWN — torch missing"


# ───────────────────────────────────────────────── 3. INTERNET (critical)
def probe_internet() -> None:
    head("3. INTERNET  (Kaggle: Settings -> Internet must be ON)")
    import urllib.request

    targets = {
        "pypi": "https://pypi.org/simple/",
        "github": "https://github.com",
        "gemini_api": "https://generativelanguage.googleapis.com/",
        "hf": "https://huggingface.co",
    }
    ok = 0
    for name, url in targets.items():
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                kv(name, f"OK (HTTP {resp.status})")
                ok += 1
        except Exception as exc:  # noqa: BLE001
            kv(name, f"FAIL — {type(exc).__name__}: {exc}")
    VERDICT["internet"] = "YES" if ok >= 2 else "NO — enable Internet in notebook settings"


# ───────────────────────────────────────────────────────── 4. PACKAGES
def probe_packages() -> None:
    head("4. PACKAGE VERSIONS")
    for mod in (
        "numpy", "cv2", "pydantic", "yaml", "tqdm", "PIL",
        "torch", "torchvision", "ultralytics", "transformers",
        "google.genai", "google.generativeai", "dotenv",
        "supervision", "timm", "addict", "yapf", "pycocotools",
    ):
        kv(mod, ver(mod))
    print("\n--- ffmpeg ---")
    print(sh("ffmpeg -version", limit=2))


# ─────────────────────────────────────────────────── 5. GROUNDING DINO
def probe_groundingdino() -> None:
    head("5. GROUNDING DINO")
    kv("groundingdino import", ver("groundingdino"))

    # The CUDA extension is the usual failure point.
    try:
        from groundingdino import _C  # noqa: F401

        kv("_C CUDA extension", "OK — compiled and importable")
        VERDICT["dino_cuda_ext"] = "YES"
    except Exception as exc:  # noqa: BLE001
        kv("_C CUDA extension", f"FAIL — {type(exc).__name__}: {exc}")
        VERDICT["dino_cuda_ext"] = "NO — needs `pip install -e` build with CUDA"

    try:
        from groundingdino.util.inference import load_model, predict  # noqa: F401

        kv("util.inference", "OK (load_model, predict importable)")
    except Exception as exc:  # noqa: BLE001
        kv("util.inference", f"FAIL — {type(exc).__name__}: {exc}")

    # Paths the current code hardcodes (groundingdino_detector.py:55-57)
    home = Path.home()
    ckpt = home / "GroundingDINO" / "groundingdino_swint_ogc.pth"
    cfg = home / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    print("\n--- paths the pipeline currently hardcodes ---")
    kv("checkpoint expected", ckpt)
    kv("  exists", ckpt.exists())
    kv("config expected", cfg)
    kv("  exists", cfg.exists())
    VERDICT["dino_paths"] = "YES" if (ckpt.exists() and cfg.exists()) else \
        "NO — place files at ~/GroundingDINO/ or pass paths via config"

    print("\n--- any .pth weights anywhere readable ---")
    print(sh("find /kaggle/input /root /kaggle/working -maxdepth 4 -name '*.pth' "
             "-size +10M 2>/dev/null | head -20"))


# ─────────────────────────────────────────────── 6. KAGGLE MOUNTS
def probe_mounts() -> None:
    head("6. /kaggle/input DATASETS  (where your uploaded videos land)")
    root = Path("/kaggle/input")
    if not root.exists():
        print("  /kaggle/input does not exist — not running on Kaggle.")
        VERDICT["datasets"] = "N/A"
        return

    entries = sorted(root.iterdir())
    if not entries:
        print("  (empty — no datasets attached to this notebook)")
        VERDICT["datasets"] = "NO — attach your video dataset via '+ Add Input'"
        return

    found_video = False
    for ds in entries:
        print(f"\n  DATASET: {ds.name}")
        try:
            for item in sorted(ds.rglob("*"))[:40]:
                if item.is_file():
                    size = human(item.stat().st_size)
                    rel = item.relative_to(ds)
                    print(f"      {size:>10}  {rel}")
                    if item.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                        found_video = True
        except Exception as exc:  # noqa: BLE001
            print(f"      (walk failed: {exc})")
    VERDICT["datasets"] = "YES (video found)" if found_video else \
        "PARTIAL — datasets attached but no video file seen"


# ─────────────────────────────────────────────────────── 7. SECRETS
def probe_secrets() -> None:
    head("7. KAGGLE SECRETS  (never prints the value)")
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            try:
                val = client.get_secret(name)
                kv(name, f"OK — retrievable, length={len(val)}")
                VERDICT["gemini_key"] = "YES"
            except Exception as exc:  # noqa: BLE001
                kv(name, f"not set ({type(exc).__name__})")
    except Exception as exc:  # noqa: BLE001
        kv("kaggle_secrets", f"unavailable ({exc})")
    VERDICT.setdefault("gemini_key", "NO — add it under Add-ons -> Secrets")

    kv("env GEMINI_API_KEY", "set" if os.environ.get("GEMINI_API_KEY") else "unset")


# ─────────────────────────────────────────────────────── 8. VIDEO IO
def probe_video_io() -> None:
    head("8. VIDEO DECODE TEST  (can OpenCV read your uploaded video?)")
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        kv("cv2", f"MISSING ({exc})")
        VERDICT["video_io"] = "NO — opencv missing"
        return

    candidates: list[Path] = []
    for base in (Path("/kaggle/input"), Path.cwd()):
        if base.exists():
            try:
                candidates += [p for p in base.rglob("*.mp4")][:5]
            except Exception:  # noqa: BLE001
                pass

    if not candidates:
        print("  no .mp4 found to test")
        VERDICT["video_io"] = "UNTESTED — no video available"
        return

    ok = False
    for video in candidates[:3]:
        cap = cv2.VideoCapture(str(video))
        opened = cap.isOpened()
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        read_ok = False
        if opened:
            read_ok, _ = cap.read()
        cap.release()
        print(f"  {video}")
        kv("    opened", opened)
        kv("    frames/fps/res", f"{frames} / {fps:.2f} / {w}x{h}")
        kv("    first frame read", read_ok)
        ok = ok or (opened and read_ok)
    VERDICT["video_io"] = "YES" if ok else "NO — OpenCV cannot decode the video"


# ─────────────────────────────────────────────────── 9. REPO / PIPELINE
def probe_repo() -> None:
    head("9. REPO + PIPELINE IMPORT")
    kv("cwd", Path.cwd())
    for name in ("src", "config", "scripts", "requirements.txt"):
        kv(f"./{name} exists", Path(name).exists())

    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.config import load_config  # noqa: F401

        kv("import src.config", "OK")
        try:
            cfg = load_config()
            kv("load_config()", "OK")
            kv("  detector.backend", getattr(cfg.detector, "backend", "(no field)"))
            kv("  detector fields", sorted(cfg.detector.model_dump().keys()))
            kv("  vlm fields", sorted(cfg.vlm.model_dump().keys()))
        except Exception as exc:  # noqa: BLE001
            kv("load_config()", f"FAIL — {exc}")
        VERDICT["pipeline_import"] = "YES"
    except Exception as exc:  # noqa: BLE001
        kv("import src.config", f"FAIL — {type(exc).__name__}: {exc}")
        print(traceback.format_exc()[-1500:])
        VERDICT["pipeline_import"] = "NO — run this from the repo root"

    print("\n--- config/ contents ---")
    cfg_dir = Path("config")
    if cfg_dir.exists():
        for f in sorted(cfg_dir.iterdir()):
            kv(f.name, human(f.stat().st_size))


# ──────────────────────────────────────────────────────── VERDICT
def print_verdict() -> None:
    head("VERDICT — can we run the pipeline here?")
    for key in ("on_kaggle", "gpu", "internet", "datasets", "video_io",
                "gemini_key", "dino_cuda_ext", "dino_paths", "pipeline_import"):
        status = VERDICT.get(key, "UNKNOWN")
        mark = "OK  " if status.startswith(("YES", "N/A")) else "!!  "
        print(f"  {mark}{key:<20} {status}")
    print(f"\n{SEP}\nPROBE COMPLETE — paste everything above back.\n{SEP}")


def main() -> int:
    print(SEP)
    print("== PHYSICAL DATA COMPILER — KAGGLE ENVIRONMENT PROBE")
    print(SEP)
    for fn in (probe_host, probe_gpu, probe_internet, probe_packages,
               probe_groundingdino, probe_mounts, probe_secrets,
               probe_video_io, probe_repo):
        try:
            fn()
        except Exception:  # noqa: BLE001
            print(f"\n!! probe section {fn.__name__} crashed:")
            print(traceback.format_exc()[-1200:])
    print_verdict()
    try:
        Path("probe_result.json").write_text(json.dumps(VERDICT, indent=2))
        print("Wrote probe_result.json")
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
