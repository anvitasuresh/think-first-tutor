"""Helpers for running the pipeline on Kaggle's free GPUs.

  python -m tutorbias.kaggle check              # which GPUs; refuses P100 (see below)
  python -m tutorbias.kaggle restore            # copy data/results from attached inputs into the repo
  python -m tutorbias.kaggle restore --dry-run

Why `restore` exists: a Kaggle session forgets everything when it ends. What survives is (a) Kaggle
Datasets you upload and (b) the /kaggle/working folder of a notebook run with "Save & Run All", which
becomes that notebook's Output. Attach either one as an Input to the next notebook and `restore`
copies its files back to where the pipeline expects them:

  .../<anything>/data/<rest>      -> data/<rest>
  .../<anything>/results/<rest>   -> results/<rest>
  items_*.jsonl, variants.jsonl, ... at a dataset's top level  -> data/processed/  (laptop upload)
  label_set.jsonl, labels_*.jsonl at a top level               -> data/labels/

When the same file comes from two inputs (or is already in the repo), the larger copy wins. Every
output in this repo is either written once or only appended to (resumable scoring), so larger means
more complete. That is what lets a stage resume from a timed-out session's partial output.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
SKIP_PARTS = {".git", "wandb", "__pycache__", ".ipynb_checkpoints"}
# Laptop uploads of data/processed often put the files at the dataset's top level.
PROCESSED_RE = re.compile(r"^(items_|messages_|variants|audit_pairs|grpo_|eval_)[\w.-]*\.(jsonl|json)$")
LABELS_RE = re.compile(r"^(label_set|label_key|labels_)[\w.-]*\.(jsonl|csv)$")


def on_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or INPUT_ROOT.exists()


def destination(rel: Path) -> Path | None:
    """Where an input file (path relative to its input root) belongs in the repo, or None to skip."""
    parts = rel.parts
    if any(p in SKIP_PARTS or p.startswith(".venv") for p in parts):
        return None
    # The *last* data/ or results/ component wins, so a nested clone (xai_proj2/data/...) maps correctly.
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in ("data", "results") and i < len(parts) - 1:
            return Path(*parts[i:])
    name = parts[-1]
    if PROCESSED_RE.match(name):
        # Keep a replies/ subfolder if the upload had one.
        return Path("data/processed", *parts[-2:]) if len(parts) > 1 and parts[-2] == "replies" else Path("data/processed", name)
    if LABELS_RE.match(name):
        return Path("data/labels", name)
    return None


def plan_restore(input_root: Path, repo: Path) -> list[tuple[Path, Path]]:
    """(source, destination) copies, keeping the largest version of each destination file."""
    best: dict[Path, Path] = {}
    if not input_root.exists():
        return []
    for src in sorted(input_root.rglob("*")):
        if not src.is_file():
            continue
        # Kaggle mounts inputs at /kaggle/input/<slug>/... (sometimes one level deeper); strip the slug.
        rel = src.relative_to(input_root)
        dst_rel = destination(Path(*rel.parts[1:])) if len(rel.parts) > 1 else None
        if dst_rel is None:
            continue
        dst = repo / dst_rel
        size = src.stat().st_size
        if dst in best and best[dst].stat().st_size >= size:
            continue
        best[dst] = src
    return [(s, d) for d, s in best.items() if not (d.exists() and d.stat().st_size >= s.stat().st_size)]


def restore(input_root: Path, repo: Path, dry_run: bool = False) -> int:
    plan = plan_restore(input_root, repo)
    for src, dst in plan:
        print(f"[restore] {src} -> {dst.relative_to(repo)}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(f"[restore] {len(plan)} files {'would be ' if dry_run else ''}copied from {input_root}")
    return len(plan)


def gpu_names() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def check_gpus(names: list[str]) -> tuple[bool, str]:
    """GPU stages need a T4 (or newer). P100s (compute capability 6.0) are not supported by vLLM and
    not by recent PyTorch CUDA builds, and have no fast fp16 tensor cores, so we stop early with a clear
    message instead of failing an hour in."""
    if not names:
        return False, "No GPU found. In the notebook sidebar: Settings > Accelerator > GPU T4 x2."
    if any("P100" in n for n in names):
        return False, ("This session has a P100, which this pipeline doesn't support (vLLM needs compute "
                       "capability >= 7.0). Switch to Settings > Accelerator > GPU T4 x2 and restart.")
    msg = f"{len(names)} GPU(s): {', '.join(names)}"
    if len(names) == 1:
        msg += " (one GPU: the 7B judge and 8B RMs need GPU T4 x2; GRPO falls back to sharing one GPU)"
    return True, msg


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["check", "restore"])
    p.add_argument("--input-root", default=str(INPUT_ROOT))
    p.add_argument("--repo", default=".")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "check":
        ok, msg = check_gpus(gpu_names())
        print(f"[kaggle] {msg}")
        sys.exit(0 if ok else 1)
    restore(Path(a.input_root), Path(a.repo).resolve(), a.dry_run)


if __name__ == "__main__":
    main()
