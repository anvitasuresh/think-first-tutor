"""The GRPO reward: the scaffolding judge, scoring each sampled tutor reply.

  reward = w_scaffolding * scaffolding  +  w_helpfulness * (helpfulness - 3) / 2

Both terms are in [-1, 1]. Nothing else goes into the reward (no length or format bonus): reward
hacking is something we want to *measure* in E6, so we don't patch it away up front.

Tutor A vs Tutor B differ only in what the judge is shown as the student's message:
  judge_sees: student   the dialect/name variant the tutor actually saw   (Tutor A, original judge)
  judge_sees: mae       the MAE base message, i.e. the judge is canonicalized (Tutor B, debiased)
The tutor itself always sees the variant. So under Tutor B, a reply's reward cannot depend on the
student's dialect by construction.

Where the judge runs. The judge is Qwen2.5-7B-Instruct, the same model and rubric as the audited
judge, but in 4-bit NF4 so it fits on the second T4 next to the policy. It runs in a child process
pinned to its own GPU (JudgeServer), which keeps the HF Trainer in the parent from seeing two GPUs and
wrapping the policy in DataParallel. The stage 2 audit also scores the 4-bit reward judge
(`SCORERS=... judge_reward`), so the exact reward used for training is itself audited.

Every scored completion is appended to <run>/rewards.jsonl with its group, so reward curves per
dialect group during training come for free (a direct look at bias propagation while it happens).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

from ..io import append_jsonl
from ..textutil import answer_leaked


def completion_text(c: Any) -> str:
    """TRL passes conversational completions as [{"role": "assistant", "content": ...}], plain ones as str."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return str(c)


def combine(scaffolding: float, helpfulness: float, w_scaff: float, w_help: float) -> float:
    return w_scaff * scaffolding + w_help * (helpfulness - 3.0) / 2.0


def judge_rows(completions: list[Any], cols: dict[str, list[Any]], judge_sees: str) -> list[dict[str, Any]]:
    """Rows in the judge's input format, one per completion. `cols` are the dataset columns TRL passes."""
    if judge_sees not in ("student", "mae"):
        raise ValueError(f"judge_sees must be 'student' or 'mae', got {judge_sees!r}")
    key = "student_message" if judge_sees == "student" else "base_message"
    return [{"question": cols["question"][i], "answer": cols["answer"][i], "student_message": cols[key][i],
             "reply": completion_text(c)} for i, c in enumerate(completions)]


# ------------------------------------------------------------------------------------- judge process
def _serve(conn: Any, spec: dict[str, Any], rubric: str, gpu: str | None) -> None:
    """Child process: load the judge on its own GPU, then score batches until told to stop."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # before torch is imported in this process
    try:
        from ..judge.judge import ScaffoldingJudge
        from ..models.backends import load_backend

        judge = ScaffoldingJudge(load_backend(spec), rubric, "logprob")
        conn.send(("ready", judge.meta))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        cmd, payload = conn.recv()
        if cmd == "close":
            return
        try:
            scores = judge.score(payload, explain=False)
            conn.send(("ok", [(s.scaffolding, s.helpfulness, s.valid_mass_scaffolding) for s in scores]))
        except Exception:
            conn.send(("error", traceback.format_exc()))


class JudgeServer:
    """Runs the judge in a spawned child process on `gpu` (a CUDA index as a string, or None)."""

    def __init__(self, spec: dict[str, Any], rubric: str = "v1", gpu: str | None = None, timeout: float = 1800):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child, spec, rubric, gpu), daemon=True)
        self.proc.start()
        if not self.conn.poll(timeout):
            raise TimeoutError("reward judge did not load in time")
        status, payload = self.conn.recv()
        if status != "ready":
            raise RuntimeError(f"reward judge failed to load:\n{payload}")
        self.meta = payload
        print(f"[reward] judge ready on GPU {gpu}: {payload}")

    def score(self, rows: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
        self.conn.send(("score", rows))
        status, payload = self.conn.recv()
        if status != "ok":
            raise RuntimeError(f"reward judge failed:\n{payload}")
        return payload

    def close(self) -> None:
        try:
            self.conn.send(("close", None))
            self.proc.join(30)
        except (BrokenPipeError, OSError):
            pass


class LocalJudge:
    """Same interface as JudgeServer, in-process. Used by tests (with a FakeBackend judge)."""

    def __init__(self, judge: Any):
        self.judge = judge
        self.meta = judge.meta

    def score(self, rows):
        return [(s.scaffolding, s.helpfulness, s.valid_mass_scaffolding) for s in self.judge.score(rows)]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------- reward function
def make_reward_func(judge: Any, rcfg: dict[str, Any], log_path: str | Path | None = None):
    """Build a TRL reward function: f(prompts, completions, **dataset_columns) -> list[float]."""
    w_s, w_h = rcfg.get("w_scaffolding", 1.0), rcfg.get("w_helpfulness", 0.5)
    judge_sees = rcfg.get("judge_sees", "student")
    state = {"calls": 0}

    def judge_reward(prompts: list[Any], completions: list[Any], **cols: Any) -> list[float]:
        t0 = time.time()
        rows = judge_rows(completions, cols, judge_sees)
        scores = judge.score(rows)
        rewards = [combine(s, h, w_s, w_h) for s, h, _ in scores]
        state["calls"] += 1
        if log_path:
            step = cols.get("trainer_state").global_step if cols.get("trainer_state") is not None else state["calls"]
            append_jsonl(log_path, (
                {"step": step, "call": state["calls"], "item_id": cols["item_id"][i], "group": cols["group"][i],
                 "scaffolding": s, "helpfulness": h, "valid_mass": m, "reward": r,
                 "answer_leaked": answer_leaked(rows[i]["reply"], str(cols["answer"][i])),
                 "n_words": len(rows[i]["reply"].split())}
                for i, ((s, h, m), r) in enumerate(zip(scores, rewards))))
        if state["calls"] % 10 == 1:
            print(f"[reward] call {state['calls']}: {len(rows)} replies in {time.time() - t0:.1f}s, "
                  f"mean reward {sum(rewards) / len(rewards):+.3f}")
        return rewards

    return judge_reward
