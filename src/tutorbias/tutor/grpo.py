"""Stage 4: train a tutor with GRPO (Shao et al., 2024), rewarded by the scaffolding judge.

  python -m tutorbias.tutor.grpo --config configs/grpo.yaml --run tutor_a            # Tutor A
  python -m tutorbias.tutor.grpo --config configs/grpo.yaml --run tutor_a --max-steps 10   # smoke test
  python -m tutorbias.tutor.grpo --config configs/grpo.yaml --run tutor_b            # stretch: debiased judge

Built for Kaggle's 2x T4 (16 GB each, no bfloat16):
  GPU 0  policy: Qwen2.5-1.5B-Instruct in fp16 + LoRA adapters kept in fp32 (fp16 mixed precision with a
         grad scaler; T4s can't do bf16). The reference model for the KL term is the same weights with
         the adapter switched off, so it costs no extra memory.
  GPU 1  reward judge: Qwen2.5-7B-Instruct in 4-bit, in its own process (see reward.py).
With one GPU, both share it (use the 0.5B policy: --model Qwen/Qwen2.5-0.5B-Instruct).

Each GRPO step samples `num_generations` replies per prompt, scores them with the judge, and pushes
the policy toward replies that beat their group's mean (group-relative advantage, no value network).

Timeouts. Kaggle sessions stop at 12 h. A checkpoint (adapter + optimizer state) is saved every
`save_steps`, and a time guard stops training cleanly at `max_hours` and saves. Rerunning the same
command resumes from the last checkpoint in results/tutor/<run>/ (restore a previous session's output
first: python -m tutorbias.kaggle restore). At the end the adapter is merged into the base weights
(results/tutor/<run>/merged) so stage 5 can load the tutor in vLLM like any other model.

TRL's GRPOConfig changes between releases, so config keys that this TRL version doesn't know are
dropped with a printed warning instead of crashing (see `filter_kwargs`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ..io import load_config, read_jsonl


def visible_gpus() -> int:
    """Count GPUs without importing torch (CUDA_VISIBLE_DEVICES must be set before torch loads)."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


def assign_gpus(n_gpus: int, policy_gpu: int = 0, judge_gpu: int | None = None) -> tuple[str | None, str | None]:
    """Policy and judge on different GPUs when there are two or more; shared otherwise."""
    if n_gpus == 0:
        return None, None
    if judge_gpu is None:
        judge_gpu = 1 if n_gpus > 1 else 0
    return str(policy_gpu), str(min(judge_gpu, n_gpus - 1))


def filter_kwargs(cls: type, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep only the keys `cls` (a dataclass such as GRPOConfig) accepts; return the dropped ones."""
    names = {f.name for f in dataclasses.fields(cls)}
    kept = {k: v for k, v in kwargs.items() if k in names}
    return kept, sorted(set(kwargs) - names)


def last_checkpoint(out_dir: Path) -> str | None:
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    # A checkpoint is complete once trainer_state.json exists (it's written last).
    ckpts = [c for c in ckpts if (c / "trainer_state.json").exists()]
    return str(ckpts[-1]) if ckpts else None


def run_config(cfg: dict[str, Any], run: str) -> dict[str, Any]:
    """Merge `runs.<run>` overrides on top of the shared settings."""
    from ..io import deep_merge

    if run not in cfg.get("runs", {}):
        raise KeyError(f"No run {run!r} in config; choose from {list(cfg.get('runs', {}))}")
    shared = {k: v for k, v in cfg.items() if k != "runs"}
    return deep_merge(shared, cfg["runs"][run] or {})


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/grpo.yaml")
    p.add_argument("--run", default="tutor_a", help="key under runs: in the config")
    p.add_argument("--model", default=None, help="override the policy model (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    p.add_argument("--max-steps", type=int, default=None, help="override max_steps (smoke test)")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args(argv)
    cfg = run_config(load_config(a.config), a.run)
    if a.model:
        cfg["policy"]["model"] = a.model
    if a.max_steps:
        cfg["train"]["max_steps"] = a.max_steps
    if a.seed is not None:
        cfg["seed"] = a.seed
    out_dir = Path(cfg["out_root"]) / a.run
    out_dir.mkdir(parents=True, exist_ok=True)

    # GPUs first, before torch is imported anywhere in this process.
    policy_gpu, judge_gpu = assign_gpus(visible_gpus(), cfg["policy"].get("gpu", 0), cfg["reward"].get("gpu"))
    if policy_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = policy_gpu
    print(f"[grpo] run={a.run} policy GPU {policy_gpu}, judge GPU {judge_gpu}, out={out_dir}")

    from .reward import JudgeServer, make_reward_func

    judge_cfg = load_config(cfg["reward"]["judge_config"])["judges"][cfg["reward"]["judge"]]
    judge = JudgeServer(judge_cfg["model"], judge_cfg.get("rubric", "v1"), judge_gpu)
    try:
        train(cfg, a.run, out_dir, judge, make_reward_func)
    finally:
        judge.close()


def train(cfg: dict[str, Any], run: str, out_dir: Path, judge: Any, make_reward_func: Any) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    set_seed(cfg["seed"])
    pol = cfg["policy"]
    fp16 = pol.get("dtype", "float16") == "float16"

    rows = read_jsonl(cfg["data"]["train_prompts"])
    keep = ("prompt", "item_id", "group", "question", "answer", "student_message", "base_message")
    ds = Dataset.from_list([{k: r[k] for k in keep} for r in rows]).shuffle(seed=cfg["seed"])
    print(f"[grpo] {len(ds)} training prompts")

    tok = AutoTokenizer.from_pretrained(pol["model"], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        pol["model"], torch_dtype=getattr(torch, pol.get("dtype", "float16")),
        attn_implementation="sdpa").to("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["train"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]))
    # fp16 training needs fp32 trainable weights (the grad scaler refuses fp16 gradients).
    for prm in model.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.print_trainable_parameters()

    targs = {**cfg["train"], "output_dir": str(out_dir), "seed": cfg["seed"], "fp16": fp16 and torch.cuda.is_available(),
             "bf16": False, "report_to": ["wandb"] if os.environ.get("WANDB_API_KEY") else [],
             "run_name": f"grpo-{run}", "remove_unused_columns": False}
    targs.pop("max_hours", None)
    gconf_kwargs, dropped = filter_kwargs(GRPOConfig, targs)
    if dropped:
        print(f"[grpo] WARNING: this TRL version ignores config keys {dropped}")
    gconf = GRPOConfig(**gconf_kwargs)

    class TimeGuard(TrainerCallback):
        """Stop and save before Kaggle's 12 h limit kills the session."""

        def __init__(self, hours: float):
            self.deadline = time.time() + hours * 3600

        def on_step_end(self, args, state, control, **kw):
            if time.time() > self.deadline:
                print(f"[grpo] time limit reached at step {state.global_step}; saving and stopping")
                control.should_save = True
                control.should_training_stop = True
            return control

    reward_fn = make_reward_func(judge, cfg["reward"], out_dir / "rewards.jsonl")
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=gconf, train_dataset=ds,
                          processing_class=tok, callbacks=[TimeGuard(cfg["train"].get("max_hours", 11))])
    resume = last_checkpoint(out_dir)
    print(f"[grpo] {'resuming from ' + resume if resume else 'starting fresh'}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out_dir / "adapter"))

    info = {"run": run, "policy": pol["model"], "reward": cfg["reward"], "judge": judge.meta,
            "global_step": trainer.state.global_step, "max_steps": gconf.max_steps,
            "finished": trainer.state.global_step >= gconf.max_steps, "lora": cfg["lora"],
            "train": cfg["train"], "seed": cfg["seed"]}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    if info["finished"] or cfg.get("merge_partial", False):
        merged = trainer.model.merge_and_unload()
        merged = merged.to(getattr(torch, pol.get("dtype", "float16")))
        merged.save_pretrained(out_dir / "merged", safe_serialization=True)
        tok.save_pretrained(out_dir / "merged")
        print(f"[grpo] merged tutor saved to {out_dir / 'merged'}")
    else:
        print(f"[grpo] stopped at step {trainer.state.global_step}/{gconf.max_steps}; rerun to resume")


if __name__ == "__main__":
    main()
