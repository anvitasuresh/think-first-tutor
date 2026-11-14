"""Stage 5, part 1: sample tutor replies on the held-out test set (E5-E7).

  python -m tutorbias.tutor.evaluate --config configs/eval.yaml --build      # test prompts, all groups
  python -m tutorbias.tutor.evaluate --config configs/eval.yaml --model base
  python -m tutorbias.tutor.evaluate --config configs/eval.yaml --model tutor_a

Every test question is asked with every version of its student message (dialect groups, paraphrase,
names), so differences between groups in one model's replies are paired by question. The base model
and each trained tutor answer the same prompts with the same sampling settings and seeds.

Output: results/eval/replies.jsonl, one row per (model, prompt), in the judge's input format, so the
judge and the held-out reward model score it with their usual commands. Checkpointed per model and
per chunk of prompts: a timeout loses at most one chunk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..io import append_jsonl, iter_jsonl, load_config, read_jsonl, stable_seed, write_jsonl
from ..models.backends import load_backend
from ..textutil import answer_leaked
from .data import build_eval_prompts


def build(cfg: dict[str, Any]) -> None:
    p = cfg["paths"]
    items = read_jsonl(p["items"])
    variants = [v for v in read_jsonl(p["variants"]) if v["split"] == "test"]
    rows = build_eval_prompts(items, variants, cfg.get("groups"))
    write_jsonl(p["prompts"], rows)
    print(f"[eval] {len(rows)} test prompts over {len({r['item_id'] for r in rows})} questions")


def model_spec(cfg: dict[str, Any], key: str) -> dict[str, Any] | None:
    spec = dict(cfg["models"][key])
    path = spec.get("model", "")
    if path.startswith("results/") and not Path(path).exists():
        print(f"[eval] skipping {key}: {path} not found (train it first)")
        return None
    return spec


def sample(cfg: dict[str, Any], key: str, limit: int | None = None) -> None:
    p, g = cfg["paths"], cfg["generation"]
    spec = model_spec(cfg, key)
    if spec is None:
        return
    out = Path(p["replies"])
    done = {r["prompt_id"] for r in iter_jsonl(out) if r["model"] == key} if out.exists() else set()
    rows = [r for r in read_jsonl(p["prompts"]) if r["prompt_id"] not in done]
    if limit:
        rows = rows[:limit]
    print(f"[eval] {key}: {len(done)} done, {len(rows)} to go")
    if not rows:
        return
    backend = load_backend(spec)
    chunk = cfg.get("flush_every", 1024)
    for i in range(0, len(rows), chunk):
        batch = rows[i : i + chunk]
        outs = backend.generate([r["prompt"] for r in batch], max_tokens=g["max_tokens"],
                                temperature=g["temperature"], top_p=g["top_p"],
                                seed=stable_seed("eval", i, base=cfg["seed"]))
        append_jsonl(out, (
            {"eval_id": f"{key}||{r['prompt_id']}", "model": key, **{k: v for k, v in r.items() if k != "prompt"},
             "reply": o[0].strip(), "n_words": len(o[0].split()), "answer_leaked": answer_leaked(o[0], r["answer"])}
            for r, o in zip(batch, outs)))
        print(f"[eval] {key}: {i + len(batch)}/{len(rows)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--model", default=None, help="key under models: in the config")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.build:
        build(cfg)
    if a.model:
        sample(cfg, a.model, a.limit)


if __name__ == "__main__":
    main()
