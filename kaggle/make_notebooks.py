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
REPO = "https://github.com/anvitasuresh/think-first-tutor"


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
[ -d think-first-tutor ] || git clone --depth 1 {REPO}
cd think-first-tutor && git pull --ff-only && git log --oneline -1
nvidia-smi --query-gpu=name,memory.total --format=csv
# Copy data/ and results/ from every attached Input (your data dataset, earlier notebooks' outputs).
PYTHONPATH=src python3 -m tutorbias.kaggle restore --repo .
PYTHONPATH=src python3 -m tutorbias.kaggle check
bash scripts/setup_local.sh gpu > /tmp/setup.log 2>&1 || (tail -30 /tmp/setup.log; exit 1)
.venv-gpu/bin/python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)" || true
echo "GPU env ready"
''')

TIDY = code('''
%%bash
# Keep the output small and self-contained: drop the venv symlinks (they point into /tmp).
cd /kaggle/working/think-first-tutor && rm -f .venv-gpu .venv-data
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
Every step is resumable. Kaggle can't attach a notebook's own output, so if a run stops early, download
the partial files from its Output tab, add them as a small dataset, attach that, and run again (`restore`
puts them back and each step continues where it stopped). GRPO has its own resume notebook (03r).

{body}
''')


# GRPO smoke test with an automatic fallback: 1.5B first, then 0.5B. The chosen run name is saved under
# results/ so the training cell (and the resume notebook, via restore) use the same one.
SMOKE = """set -o pipefail
mkdir -p results/tutor
if [ -f results/tutor/RUN_NAME ]; then echo "Smoke test already passed: $(cat results/tutor/RUN_NAME)"; exit 0; fi
rm -rf results/tutor/smoke
if bash scripts/run_stage.sh grpo-smoke > /tmp/smoke15.log 2>&1; then
  echo tutor_a > results/tutor/RUN_NAME; echo "SMOKE OK with 1.5B"; grep -o "'step_time': '[0-9.]*'" /tmp/smoke15.log | tail -3
else
  tail -25 /tmp/smoke15.log; echo "1.5B smoke test failed (above). Trying the 0.5B policy..."
  rm -rf results/tutor/smoke
  if POLICY=Qwen/Qwen2.5-0.5B-Instruct bash scripts/run_stage.sh grpo-smoke > /tmp/smoke05.log 2>&1; then
    echo tutor_a_0.5b > results/tutor/RUN_NAME; echo "SMOKE OK with 0.5B"
  else
    tail -40 /tmp/smoke05.log; echo "Both smoke tests failed: send the log above."; exit 1
  fi
fi
rm -rf results/tutor/smoke   # smoke weights aren't needed"""

TRAIN = """set -o pipefail
R=$(cat results/tutor/RUN_NAME)
echo "Training run: $R"
RUN=$R bash scripts/run_stage.sh grpo 2>&1 | grep -v "Warning" | tail -80
ls results/tutor/$R && cat results/tutor/$R/run_info.json"""


def stage(cmd: str) -> dict:
    return code(f"%%bash\ncd /kaggle/working/think-first-tutor\n{cmd}")


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
    "01b_rescore_judge.ipynb": [
        settings("1b. Re-score the label set with a revised rubric (E1 iteration)", "~15 min",
                 ["**tutorbias-data**"],
                 "Only needed if the judge missed the E1 gate on dev. Scores the 300 label-set replies with "
                 "a revised rubric (no rationales). Set RUBRIC in the cell (v2, v3, ...). Output: `results/judge/label_set_scores_<RUBRIC>.jsonl`."),
        SECRETS, SETUP,
        stage("export RUBRIC=v3\nEXTRA= OUTPUT=results/judge/label_set_scores_$RUBRIC.jsonl bash scripts/run_stage.sh judge"),
        TIDY,
    ],
    "02_audit.ipynb": [
        settings("2. Reward-model audit (stage 2, E2)", "~4-5 h",
                 ["**tutorbias-data**", "the Output of notebook **01_replies_judge**"],
                 "Run only after the judge passed the E1 gate on your laptop (labels committed to the repo). "
                 "Scorers: the judge (fp16, vLLM on both T4s), Skywork-Reward-V2 and Tulu-3-8B-RM (fp16, split over "
                 "both T4s), and the 4-bit GRPO reward judge. Results: `results/audit/summary.md`."),
        SECRETS, SETUP,
        stage("bash scripts/run_stage.sh audit"),
        stage("cat results/audit/summary.md"),
        TIDY,
    ],
    "03_grpo_tutor_a.ipynb": [
        settings("3. GRPO Tutor A (stage 4)", "~20 min smoke test + ~3-4 h training",
                 ["**tutorbias-data**"],
                 "Policy: Qwen2.5-1.5B-Instruct + LoRA in fp16 on GPU 0. Reward: the 7B judge in 4-bit on GPU 1. "
                 "Cell 4 is a 10-step smoke test. If 1.5B fails (e.g. out of memory) it retries with the 0.5B "
                 "policy and the real run uses that; if both fail, the notebook stops after ~20 min instead of "
                 "wasting hours. Checkpoints every 25 steps; training stops itself at 11 h. If the session dies, "
                 "run **03r_grpo_resume** with this notebook's Output attached."),
        SECRETS, SETUP,
        stage("bash scripts/run_stage.sh grpo-data"),
        stage(SMOKE),
        stage(TRAIN),
        TIDY,
    ],
    "03r_grpo_resume.ipynb": [
        settings("3r. Resume GRPO Tutor A", "whatever is left of ~3-4 h",
                 ["**tutorbias-data**", "the Output of notebook **03_grpo_tutor_a** (or of the last 03r run)"],
                 "Only if 03_grpo_tutor_a stopped before `[grpo] merged tutor saved`. Picks up the run name the "
                 "smoke test chose and the newest complete checkpoint, then trains to the end and merges."),
        SECRETS, SETUP,
        stage(TRAIN),
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
