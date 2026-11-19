# Running the pipeline on Kaggle (free)

Every GPU stage runs on Kaggle's free tier: about **30 GPU-hours a week** on **2x NVIDIA T4** (16 GB
each, no bfloat16), sessions up to 12 hours. The whole project needs roughly 14 GPU-hours. CPU stages
(items, dialect variants, labeling, calibration) run on a laptop.

| Notebook (in `kaggle/`) | Stage | Needs as Input | GPU time |
|---|---|---|---|
| `01_replies_judge.ipynb` | 1: reply pool, label set, judge scores | your data dataset | ~1.5 h |
| `01b_rescore_judge.ipynb` | E1 iteration: re-score the label set with rubric v2 (only if the gate was missed on dev) | your data dataset | ~15 min |
| `02_audit.ipynb` | 2: reward-model audit (E2) | data + output of 01 | ~4-5 h |
| `03_grpo_tutor_a.ipynb` | 4: GRPO smoke test, then Tutor A | data | ~0.25 + 3-4 h |
| `03r_grpo_resume.ipynb` | 4: resume Tutor A if 03 stopped early | data + output of 03 | what's left |
| `04_explain.ipynb` | 3: feature effects, rationale faithfulness, probes (E3-E4) | data + output of 02 | ~1-1.5 h |
| `05_eval.ipynb` | 5: base vs tutors on held-out questions (E5-E7) | data + output of 03 | ~1.5 h |

Each notebook is the same few cells: load secrets, clone this repo, copy inputs into place
(`restore`), install the GPU env into `/tmp`, run `bash scripts/run_stage.sh <stage>`. Kaggle is the
default target, so no flags are needed (`PROFILE=t4` is the default).

## One-time setup
1. Create a Kaggle account and **verify your phone number** (Settings > Phone verification). Without it
   notebooks get no GPU and no internet.
2. Optional secrets (any notebook: Add-ons > Secrets): `HF_TOKEN` (a free Hugging Face read token;
   avoids download rate limits) and `WANDB_API_KEY` (run tracking). Turn each secret on for the notebook.
3. On your laptop, after `bash scripts/run_stage.sh items` and `variants`: upload the folder
   `data/processed` as a **private Kaggle Dataset** named `tutorbias-data` (Datasets > New Dataset,
   drag the folder in). Variants are never rebuilt on GPU time.
4. Import the notebooks: Create > New Notebook > File > Import Notebook, and pick the `.ipynb` from
   `kaggle/` (upload it, or paste its GitHub URL).

## Running a notebook
Settings in the right sidebar: **Accelerator = GPU T4 x2**, **Internet = on**, and add the Inputs
listed at the top of the notebook (Add Input > Datasets > Your Datasets, or Add Input > Notebooks >
Your Work for an earlier notebook's output). Then **Save Version > Save & Run All (Commit)**. A
committed run keeps going with the tab closed, and when it ends `/kaggle/working` becomes the
notebook's **Output**, which later notebooks attach as an Input.

To try a stage quickly first, run interactively with `LIMIT=50 bash scripts/run_stage.sh <stage>`.

## Timeouts and resuming
Nothing is lost when a session stops early:
- Judge and reward-model scores are appended every batch (512 rows); reruns skip scored rows.
- Reply generation is saved per style; tutor evaluation per 1024 prompts.
- GRPO saves a checkpoint (LoRA adapter + optimizer) every 25 steps and stops itself at 11 h.

To resume: open the same notebook, **add its own previous Output as an extra Input**, and run it
again. `restore` copies the partial files back (when two copies of a file exist, the larger one wins,
since every output only grows), and each step continues from where it stopped.

## Memory plan on 2x T4
- Judge (Qwen2.5-7B-Instruct, fp16, ~15 GB): vLLM tensor parallel over both T4s.
- Reward models (8B, fp16, ~16 GB): transformers `device_map="auto"` splits them over both T4s. Not
  quantized, so the audited RMs are exactly the published models. (`load_in_4bit: true` in
  `configs/t4/audit.yaml` exists only as a fallback, and the score rows then say "(nf4)".)
- GRPO: policy (Qwen2.5-1.5B + LoRA, fp16, ~8 GB with activations) on GPU 0; reward judge (same 7B,
  same rubric, **4-bit NF4**, ~5.5 GB) on GPU 1 in its own process. The audit also scores this 4-bit
  judge, so the exact reward used for training is audited and its agreement with the fp16 judge is
  reported.

## If something breaks
- **P100 assigned**: the setup cell stops with a message. Switch the accelerator to GPU T4 x2.
- **vLLM fails to start on T4**: rerun the cell with `TUTORBIAS_BACKEND=hf` in front of the command.
  The same models then run through transformers (same scores up to numerical noise, slower generation).
- **GRPO out of memory, or slower than ~60 s/step in the smoke test**: in the training cell use
  `POLICY=Qwen/Qwen2.5-0.5B-Instruct` (or `RUN=tutor_a_0.5b`). Second option: set
  `gradient_checkpointing: true` in `configs/grpo.yaml` (less memory, slower sampling).
- **Disk full in /kaggle/working**: model weights and envs live in `/tmp`; only `data/` and
  `results/` should be in the output. Delete `results/tutor/smoke/`.

## Reporting
T4s don't support bfloat16, so every model runs in float16. The judge model and rubric are unchanged,
but the README states "fp16 on T4" for all reported numbers.
