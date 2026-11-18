"""Score audit pairs with off-the-shelf reward models from RewardBench.

  python -m tutorbias.audit.reward_models --config configs/audit.yaml --rm skywork_v2
  # any other rows (e.g. tutor replies in stage 5, as a held-out scorer):
  python -m tutorbias.audit.reward_models --rm skywork_v2 --input X.jsonl --output Y.jsonl --id-field eval_id

These are the kind of reward models people plug into RLHF pipelines. They are *audited only* (never
used for training here). Each RM sees the same conversation the tutor saw:
  system: tutor role   user: problem + student message   assistant: tutor reply
and returns one scalar reward. Scores are resumable, flushed per batch, like the judge.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from ..io import append_jsonl, iter_jsonl, load_config, sha256_file
from ..replies.generate import tutor_conv
from ..tracking import init_run


def rm_conversation(row: dict[str, Any]) -> list[dict[str, str]]:
    """Tutor conversation without a style instruction, plus the reply as the assistant turn."""
    return tutor_conv(row["question"], row["student_message"], style=None) + [
        {"role": "assistant", "content": row["reply"]}
    ]


class SequenceClassifierRM:
    """Reward models exposed as AutoModelForSequenceClassification with one output logit
    (Skywork-Reward, Tulu-3-RM, most RewardBench classifiers). Remote-code RMs (e.g. ArmoRM) can set `score_attr`."""

    def __init__(self, model: str, max_length: int = 4096, batch_size: int = 8, trust_remote_code: bool = False,
                 score_attr: str | None = None, dtype: str = "bfloat16", load_in_4bit: bool = False, **_: Any):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.name = model
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        kwargs: dict[str, Any] = {}
        if load_in_4bit:
            # Fallback only (one 16 GB GPU, or OOM). A 4-bit RM is not exactly the model we claim to audit,
            # so the name in every score row says so, and the README must report it.
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype))
            self.name = f"{model} (nf4)"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            # device_map="auto" splits an 8B model across Kaggle's 2x T4 (16 GB of fp16 weights over 30 GB).
            # T4s have no bfloat16, so the t4 profile sets dtype: float16.
            model, torch_dtype=getattr(torch, dtype), device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=trust_remote_code, attn_implementation="sdpa", **kwargs,
        ).eval()
        # Batched sequence classifiers find each row's last real token through config.pad_token_id. Some RMs
        # (e.g. Tulu-3-RM) ship a tokenizer pad token but leave the config's unset, which breaks batch > 1.
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tok.pad_token_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.score_attr = score_attr

    def score(self, rows: list[dict[str, Any]]) -> list[float]:
        texts = [self.tok.apply_chat_template(rm_conversation(r), tokenize=False) for r in rows]
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i : i + self.batch_size], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length, add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                res = self.model(**enc)
            vals = getattr(res, self.score_attr) if self.score_attr else res.logits[:, 0]
            out += vals.float().reshape(-1).cpu().tolist()
        return out


def check_batching(rm: SequenceClassifierRM, rows: list[dict[str, Any]], tol: float = 0.05) -> None:
    """Padding bugs in reward models silently change scores (the RM reads the wrong last token).
    Score a few rows batched and one at a time; stop if they disagree."""
    if rm.batch_size == 1 or len(rows) < 2:
        return
    batched = rm.score(rows)
    bs, rm.batch_size = rm.batch_size, 1
    single = rm.score(rows)
    rm.batch_size = bs
    worst = max(abs(a - b) for a, b in zip(batched, single))
    if worst > tol:
        raise RuntimeError(f"{rm.name}: batched vs single scores differ by {worst:.3f}. "
                           "Set batch_size: 1 for this RM in configs/audit.yaml.")
    print(f"[rm] batching check passed (max diff {worst:.4f})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/audit.yaml")
    p.add_argument("--rm", required=True, help="key under reward_models in the config")
    p.add_argument("--input", default=None, help="rows to score (default: the audit pairs)")
    p.add_argument("--output", default=None, help="default: <scores_dir>/<rm>.jsonl")
    p.add_argument("--id-field", default="pair_id")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    spec = cfg["reward_models"][a.rm]
    src = a.input or cfg["paths"]["pairs"]
    out = Path(a.output or Path(cfg["paths"]["scores_dir"]) / f"{a.rm}.jsonl")

    # Resumable: scores are appended per chunk (flush_every rows) and finished ids are skipped on rerun.
    done = {r["id"] for r in iter_jsonl(out)} if out.exists() else set()
    rows = [r for r in iter_jsonl(src) if r[a.id_field] not in done]
    if a.limit:
        rows = rows[: a.limit]
    print(f"[rm:{a.rm}] {len(done)} done, {len(rows)} to go")
    if not rows:
        return
    run = init_run("audit-rm", {**cfg, "rm": a.rm, "input_sha": sha256_file(src)})
    rm = SequenceClassifierRM(**spec)
    check_batching(rm, rows[:8])
    # Sort by length so batches pad less; output order doesn't matter (keyed by id).
    rows.sort(key=lambda r: len(r["reply"]) + len(r["student_message"]))
    chunk, t0 = cfg.get("flush_every", 512), time.time()
    for i in range(0, len(rows), chunk):
        batch = rows[i : i + chunk]
        scores = rm.score(batch)
        append_jsonl(out, ({"id": r[a.id_field], "scorer": a.rm, "model": rm.name, "reward": s}
                           for r, s in zip(batch, scores)))
        n = i + len(batch)
        run.log({"scored": n, "rows_per_sec": n / (time.time() - t0)})
        print(f"[rm:{a.rm}] {n}/{len(rows)}  {n / (time.time() - t0):.1f} rows/s")
    run.finish()


if __name__ == "__main__":
    main()
