"""Write the Kaggle notebooks in this folder. They are generated so every notebook shares the same
setup cells; edit this file, not the .ipynb files, then run:  python kaggle/make_notebooks.py

Each notebook: settings checklist -> secrets -> clone + restore + install -> stage commands -> tidy.
Run them with "Save Version > Save & Run All (Commit)" so they keep running with the browser closed and
/kaggle/working becomes the notebook's Output, which the next notebook attaches as an Input.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = "https://github.com/anvitasuresh/xai_proj2"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


SECRETS = code('''
# Secrets (sidebar: Add-ons > Secrets). HF_TOKEN is optional (the models are public, but a token avoids
# download rate limits); WANDB_API_KEY is optional (without it, metrics are only written to files).
import os
try:
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    for k in ("HF_TOKEN", "WANDB_API_KEY"):
        try:
            os.environ[k] = s.get_secret(k)
            print(k, "loaded")
        except Exception:
            print(k, "not set (fine)")
except ImportError:
    print("not on Kaggle")
# Big files go to /tmp (the 20 GB /kaggle/working is saved as the notebook's output; keep it small).
os.environ.update(HF_HOME="/tmp/hf", VENV_ROOT="/tmp", UV_CACHE_DIR="/tmp/uv", PYTHONUNBUFFERED="1")
''')

SETUP = code(f'''
%%bash
set -e
cd /kaggle/working
[ -d xai_proj2 ] || git clone --depth 1 {REPO}
cd xai_proj2 && git pull --ff-only && git log --oneline -1
nvidia-smi --query-gpu=name,memory.total --format=csv
# Copy data/ and results/ from every attached Input (your data dataset, earlier notebooks' outputs).
PYTHONPATH=src python3 -m tutorbias.kaggle restore --repo .
PYTHONPATH=src python3 -m tutorbias.kaggle check
bash scripts/setup_local.sh gpu > /tmp/setup.log 2>&1 || (tail -30 /tmp/setup.log; exit 1)
echo "GPU env ready"
''')

TIDY = code('''
%%bash
# Keep the output small and self-contained: drop the venv symlinks (they point into /tmp).
cd /kaggle/working/xai_proj2 && rm -f .venv-gpu .venv-data
du -sh data results 2>/dev/null || true
''')


def settings(title: str, minutes: str, inputs: list[str], body: str) -> dict:
    ins = "\n".join(f"   - {i}" for i in inputs) or "   - none"
    return md(f'''
# {title}

**Settings (right sidebar), before running:**
1. Accelerator: **GPU T4 x2** (not P100: this pipeline refuses it).
2. Internet: **on** (needs a phone-verified account).
3. Add Input:
{ins}

Run with **Save Version > Save & Run All (Commit)** so it survives closing the tab. Expected GPU time: **{minutes}**.
Every step is resumable: if the run stops early, attach this notebook's own output as an extra Input
and run it again; `restore` picks up the partial files and each step continues where it stopped.

{body}
''')


def stage(cmd: str) -> dict:
    return code(f"%%bash\ncd /kaggle/working/xai_proj2\n{cmd}")


NOTEBOOKS = {
    "01_replies_judge.ipynb": [
        settings("1. Reply pool + judge scores (stage 1)", "~1.5 h",
                 ["your dataset **tutorbias-data** (the `data/processed` folder from your laptop)"],
                 "Output you need afterwards, for labeling on your laptop: `data/labels/label_set.jsonl` and "
                 "`results/judge/label_set_scores.jsonl` (download from this notebook's Output tab)."),
        SECRETS, SETUP,
        stage("bash scripts/run_stage.sh replies   # 1.5B + 7B reply pool, label set"),
        stage("bash scripts/run_stage.sh judge     # judge scores + rationales on the label set"),
        TIDY,
    ],
    "02_audit.ipynb": [
        settings("2. Reward-model audit (stage 2, E2)", "~4-5 h",
                 ["**tutorbias-data**", "the Output of notebook **01_replies_judge**"],
                 "Run only after the judge passed the E1 gate on your laptop (labels committed to the repo). "
                 "Scorers: the judge (fp16, vLLM on both T4s), Skywork-Reward-V2 and ArmoRM (fp16, split over "
                 "both T4s), and the 4-bit GRPO reward judge. Results: `results/audit/summary.md`."),
        SECRETS, SETUP,
        stage("bash scripts/run_stage.sh audit"),
        stage("cat results/audit/summary.md"),
        TIDY,
    ],
    "03_grpo_tutor_a.ipynb": [
        settings("3. GRPO Tutor A (stage 4)", "~15 min smoke + ~3-4 h training",
                 ["**tutorbias-data**", "(when resuming) this notebook's own previous Output"],
                 "Policy: Qwen2.5-1.5B-Instruct + LoRA in fp16 on GPU 0. Reward: the 7B judge in 4-bit on GPU 1. "
                 "The smoke test prints seconds per step; if it runs out of memory or is slower than ~60 s/step, "
                 "set `POLICY=Qwen/Qwen2.5-0.5B-Instruct` in the training cell. Checkpoints every 25 steps; "
                 "training stops itself at 11 h."),
        SECRETS, SETUP,
        stage("bash scripts/run_stage.sh grpo-data"),
        stage("bash scripts/run_stage.sh grpo-smoke 2>&1 | tail -60\nrm -rf results/tutor/smoke/merged   # smoke weights aren't needed"),
        stage("RUN=tutor_a bash scripts/run_stage.sh grpo   # add POLICY=Qwen/Qwen2.5-0.5B-Instruct if needed"),
        stage("ls results/tutor/tutor_a && cat results/tutor/tutor_a/run_info.json"),
        TIDY,
    ],
    "04_explain.ipynb": [
        settings("4. Explain the gaps (stage 3, E3-E4)", "~1-1.5 h",
                 ["**tutorbias-data**", "the Output of notebook **02_audit**"],
                 "Feature effects from the audit scores, judge rationales on a 200-reply subset (faithfulness), "
                 "and linear probes on the judge's activations. Results in `results/explain/`."),
        SECRETS, SETUP,
        stage("bash scripts/run_stage.sh explain"),
        stage("cat results/explain/faithfulness.md results/explain/probes.md"),
        TIDY,
    ],
    "05_eval.ipynb": [
        settings("5. Evaluate the tutors (stage 5, E5-E7)", "~1.5 h",
                 ["**tutorbias-data**", "the Output of notebook **03_grpo_tutor_a** (needs `results/tutor/tutor_a/merged`)"],
                 "Base model vs Tutor A (and Tutor B if trained) on held-out questions, every student version. "
                 "Judge + held-out Skywork RM. Results: `results/eval/summary.md`, `e7_did.csv`."),
        SECRETS, SETUP,
        stage("bash scripts/run_stage.sh eval"),
        stage("cat results/eval/summary.md"),
        TIDY,
    ],
}


def main() -> None:
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
    for name, cells in NOTEBOOKS.items():
        nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
        (HERE / name).write_text(json.dumps(nb, indent=1) + "\n")
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
