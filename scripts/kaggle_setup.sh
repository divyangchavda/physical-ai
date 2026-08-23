#!/usr/bin/env bash
# Kaggle session bootstrap for the Physical Data Compiler.
#
# Run ONCE per Kaggle session, from the repo root:
#     !bash scripts/kaggle_setup.sh
#
# Design rule: NEVER let pip touch Kaggle's preinstalled CUDA torch.
# A plain `pip install ultralytics` can pull a different (or +cpu) torch and
# silently kill GPU support. We pin the installed torch via PIP_CONSTRAINT so
# pip is not allowed to move it.

set -uo pipefail   # not -e: a soft failure should still print a diagnosis

echo "=============================================================="
echo "  PDC KAGGLE SETUP"
echo "=============================================================="

# ── 1. Pin the existing torch so nothing can replace it ────────────────────
python - <<'PY'
from pathlib import Path
try:
    import torch, torchvision
    Path("/tmp/pdc_constraints.txt").write_text(
        f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n"
    )
    print(f"[pin] torch=={torch.__version__}  torchvision=={torchvision.__version__}")
    print(f"[pin] cuda available: {torch.cuda.is_available()}")
except Exception as exc:
    Path("/tmp/pdc_constraints.txt").write_text("")
    print(f"[pin] WARNING: torch not importable ({exc}); no constraint written")
PY
export PIP_CONSTRAINT=/tmp/pdc_constraints.txt

# ── 2. Runtime deps (constrained; skips anything already present) ──────────
echo ""
echo "--- installing pipeline deps ---"
pip install -q --no-warn-conflicts \
    "numpy>=1.24" "opencv-python>=4.8" "pydantic>=2.5" "PyYAML>=6.0" "tqdm>=4.66" \
    || echo "!! core dep install reported problems"

pip install -q --no-warn-conflicts "ultralytics==8.4.120" \
    || echo "!! ultralytics install reported problems"

pip install -q --no-warn-conflicts "google-genai" "python-dotenv" \
    || echo "!! gemini/dotenv install reported problems"

# ── 3. GroundingDINO source (needed for the .py model config + package) ────
echo ""
echo "--- GroundingDINO ---"
DINO_DIR=/kaggle/working/GroundingDINO
if [ ! -d "$DINO_DIR" ]; then
    git clone -q https://github.com/IDEA-Research/GroundingDINO.git "$DINO_DIR" \
        && echo "[dino] cloned to $DINO_DIR" \
        || echo "!! clone failed — is Internet enabled in notebook settings?"
else
    echo "[dino] already present at $DINO_DIR"
fi

# CUDA_HOME must be set or the custom op silently builds CPU-only / fails.
if [ -z "${CUDA_HOME:-}" ]; then
    for c in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11; do
        [ -d "$c" ] && export CUDA_HOME="$c" && break
    done
fi
echo "[dino] CUDA_HOME=${CUDA_HOME:-<unset>}"

if [ -d "$DINO_DIR" ]; then
    echo "[dino] building (compiles the MultiScaleDeformableAttention CUDA op; 3-5 min first time)"
    # --no-build-isolation is REQUIRED: with isolation, pip builds against a
    # freshly downloaded torch instead of Kaggle's, and the op fails to load.
    (cd "$DINO_DIR" && pip install -q -e . --no-build-isolation) \
        && echo "[dino] pip install -e OK" \
        || echo "!! GroundingDINO build FAILED — see plan B in KAGGLE.md"
fi

# ── 4. Verify — this is the part that actually matters ─────────────────────
echo ""
echo "--- verification ---"
python - <<'PY'
import importlib, os
from pathlib import Path

def check(label, fn):
    try:
        print(f"  OK   {label}: {fn()}")
        return True
    except Exception as exc:
        print(f"  FAIL {label}: {type(exc).__name__}: {exc}")
        return False

import_ok = check("torch", lambda: __import__("torch").__version__)
check("torch.cuda.is_available", lambda: __import__("torch").cuda.is_available())
check("cv2", lambda: importlib.import_module("cv2").__version__)
check("pydantic", lambda: importlib.import_module("pydantic").__version__)
check("ultralytics", lambda: importlib.import_module("ultralytics").__version__)
check("groundingdino.util.inference",
      lambda: bool(importlib.import_module("groundingdino.util.inference").load_model))

# The single most common Kaggle failure point:
try:
    from groundingdino import _C  # noqa: F401
    print("  OK   groundingdino._C CUDA extension importable")
except Exception as exc:
    print(f"  FAIL groundingdino._C: {type(exc).__name__}: {exc}")
    print("       -> GroundingDINO will not run. See 'Plan B' in KAGGLE.md.")

cfg = Path("/kaggle/working/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(f"  {'OK  ' if cfg.exists() else 'FAIL'} model config present: {cfg}")

ckpts = list(Path("/kaggle/input").rglob("*.pth")) if Path("/kaggle/input").exists() else []
if ckpts:
    for c in ckpts[:5]:
        print(f"  OK   checkpoint found: {c}  ({c.stat().st_size / 1e6:.0f} MB)")
else:
    print("  FAIL no .pth checkpoint under /kaggle/input")
    print("       -> upload groundingdino_swint_ogc.pth as a Kaggle Dataset,")
    print("          or run the download cell in KAGGLE.md")
PY

echo ""
echo "=============================================================="
echo "  SETUP DONE — next: python scripts/kaggle_probe.py"
echo "=============================================================="
