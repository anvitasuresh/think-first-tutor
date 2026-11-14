# Legacy: Duke DCC / SLURM

The project originally planned to run on Duke's DCC cluster. Access never came through, so as of
Nov 14, 2026 everything runs on Kaggle's free 2x T4 instead (see [docs/kaggle.md](../docs/kaggle.md)).

These SLURM scripts are kept for anyone with a SLURM cluster. They only wrap `scripts/run_stage.sh`
(stages 1-2) and are no longer maintained or tested. On a 24 GB+ Ampere GPU, run any stage with
`PROFILE=none bash scripts/run_stage.sh <stage>` to use the bf16 base configs.
