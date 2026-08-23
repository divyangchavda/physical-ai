# Kaggle Workflow — code locally, test on GPU

Code on Windows → push to GitHub → pull on Kaggle → run on GPU → download results.
Nothing large ever goes through git.

---

## 1. Repo layout (after the restructure)

```
physical-ai/                    ← repo root, was AI-object-/
├── src/                        pipeline code (14 stages)
├── config/                     experiment YAMLs
│   ├── default.yaml            base config, all defaults
│   ├── smoke.yaml              30-second sanity run — use before spending quota
│   ├── kaggle_tt6_dino.yaml    GroundingDINO w/ absolute Kaggle paths
│   └── tt*.yaml                past experiments
├── scripts/
│   ├── kaggle_probe.py         environment recon — run this first
│   ├── kaggle_setup.sh         one-shot session bootstrap
│   ├── dump_run.py             print every output file + its JSON
│   └── phase*_smoke.py         per-stage smoke tests
├── tools/                      one-off analysis scripts (not part of pipeline)
├── tests/                      pytest suite
├── docs/                       experiment reports + extracted spec
│
├── data/videos/                GITIGNORED — local videos, uploaded to Kaggle manually
├── weights/                    GITIGNORED — *.pt / *.pth
├── runs/                       GITIGNORED — all pipeline output
├── .env                        GITIGNORED — GEMINI_API_KEY
└── .env.example                template, committed
```

**What git carries:** 127 source files, a few hundred KB. Clones in seconds on the
GPU clock. Videos, weights and outputs are all excluded.

---

## 2. Kaggle layout

```
/kaggle/input/                  read-only, persistent, free, survives sessions
  pdc-videos/                     ← YOUR UPLOADED VIDEO DATASET
    tt6.mp4
    tt3.mp4
  pdc-weights/                    ← YOUR UPLOADED WEIGHTS DATASET
    groundingdino_swint_ogc.pth
    yolov8n.pt

/kaggle/working/                writable, ~20 GB, WIPED when session ends
  physical-ai/                    ← git clone lands here (the repo)
  GroundingDINO/                  ← cloned source, needed for its .py model config
  runs/tt6_dino/                  ← pipeline output: pipeline.log + all JSON
  results.zip                     ← download this before the session dies
```

The critical asymmetry: **`/kaggle/input` persists, `/kaggle/working` does not.**
Upload once to `input`; treat everything in `working` as disposable.

---

## 3. Where to upload the video — do this once

1. Go to **kaggle.com/datasets** → **New Dataset**
2. Drag in your videos from `data/videos/` (start with just `tt6.mp4`, 500 MB —
   don't upload all 10 while testing the workflow)
3. Title it exactly **`pdc-videos`** → **Create**
4. Repeat for weights: New Dataset titled **`pdc-weights`**, containing
   `groundingdino_swint_ogc.pth` and `yolov8n.pt`
5. In your notebook: **+ Add Input** → search your datasets → add both

It mounts at `/kaggle/input/pdc-videos/tt6.mp4`. Run `scripts/kaggle_probe.py`
to see the exact mount path — Kaggle sometimes slugifies the name.

To add a video later: open the dataset → **New Version** → drag → the notebook
picks it up on next run. No re-upload of existing files.

**Getting the GroundingDINO checkpoint** (660 MB — you likely don't have it locally):
download it directly on Kaggle once, then save it as a Dataset output so you
never download it again:

```python
!wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O /kaggle/working/groundingdino_swint_ogc.pth
```

---

## 4. How the code finds the Kaggle files

Two mechanisms, no code editing per environment:

**Video** — a positional CLI argument:
```bash
python -m src.pipeline /kaggle/input/pdc-videos/tt6.mp4
```

**Weights** — config fields (`DetectorConfig.model_checkpoint` / `config_file`).
`config/kaggle_tt6_dino.yaml` sets them to absolute Kaggle paths:

```yaml
detector:
  model_checkpoint: "/kaggle/input/pdc-weights/groundingdino_swint_ogc.pth"
  config_file: "/kaggle/working/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
```

When these are `null` (the local case) the detector falls back to
`~/GroundingDINO/`. Same code, both environments.

---

## 5. Notebook — five cells

Settings first: **Accelerator = GPU T4 x2**, **Internet = ON**
(without Internet, `git clone`, `pip`, and the Gemini API all fail).

**Cell 1 — get the code**
```python
!rm -rf /kaggle/working/physical-ai
!git clone -q https://github.com/divyangchavda/physical-ai.git /kaggle/working/physical-ai
%cd /kaggle/working/physical-ai
!git log --oneline -1
```

**Cell 2 — bootstrap the session** (~4 min, once per session)
```python
!bash scripts/kaggle_setup.sh
```

**Cell 3 — probe the environment, then paste this output back**
```python
!python scripts/kaggle_probe.py
```

**Cell 4 — API key from Kaggle Secrets** (skip while `vlm.enabled: false`)

Add-ons → Secrets → **Add a new secret**, name it exactly `GEMINI_API_KEY`,
paste the value, and tick the checkbox to attach it to this notebook.

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ['GEMINI_API_KEY'] = UserSecretsClient().get_secret('GEMINI_API_KEY')
print('key set, length:', len(os.environ['GEMINI_API_KEY']))
```

`os.environ` set in a notebook cell is inherited by every later `!command`,
so the pipeline sees it. Never write the key into a `.py` or `.yaml` file.

**Cell 5 — smoke test before spending quota** (~30 s)
```python
!python -m src.pipeline /kaggle/input/pdc-videos/tt3.mp4 \
    --config config/smoke.yaml \
    --output-dir /kaggle/working/runs/smoke --verbose
```

If that reaches `s14_preview`, the whole code path is proven. Then the real run:

**Cell 6 — full run, log captured to disk**
```python
!python -m src.pipeline /kaggle/input/pdc-videos/tt6.mp4 \
    --config config/kaggle_tt6_dino.yaml \
    --output-dir /kaggle/working/runs/tt6_dino \
    --verbose 2>&1 | tee /kaggle/working/runs/tt6_dino/run.log
```

**Cell 7 — dump every output file + JSON, then package it**
```python
!python scripts/dump_run.py /kaggle/working/runs/tt6_dino 2>&1 | tee /kaggle/working/dump.txt
!cd /kaggle/working && zip -qr results.zip runs/ dump.txt && ls -lh results.zip
```

Download `results.zip` from the notebook's Output panel. That zip plus the
printed dump is everything needed to diagnose the run.

---

## 6. Plan B if the GroundingDINO CUDA op fails to build

`from groundingdino import _C` failing is the most likely blocker — the custom
`MultiScaleDeformableAttention` op has to compile against the exact torch/CUDA
pair Kaggle ships, and it breaks on new torch versions.

If the probe reports `dino_cuda_ext: NO`, the fix is to stop compiling anything
and use the HuggingFace port instead, which is pure PyTorch:

```python
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
model_id = "IDEA-Research/grounding-dino-base"
```

That needs a new `src/models/hf_groundingdino_detector.py` behind the same
`ObjectDetector` interface — roughly 80 lines, no build step, and it removes the
660 MB checkpoint upload entirely. Report the probe output and this can be
written if needed.

---

## 7. Quota discipline

GPU is ~30 h/week and the pipeline has **no resume flag** — a crash in stage 11
re-runs detection from scratch. So:

- Always run `config/smoke.yaml` first. It costs 30 s and catches most crashes.
- Keep `vlm.enabled: false` while working on detection/tracking. Gemini costs
  ~220 s *per segment* and its output does not depend on detections.
- **Stop the session manually** when done. An idle notebook keeps burning quota.
- Commit nothing from Kaggle; the repo is the source of truth in one direction only.

---

## 8. Daily loop

```
edit locally  →  git add -A && git commit && git push
              →  Kaggle Cell 1 (re-clone)  →  Cell 5 smoke  →  Cell 6 run
              →  Cell 7 dump + zip  →  download  →  analyse  →  fix locally
```

Re-running Cell 1 is a fresh clone, so there is never a merge conflict on Kaggle.
