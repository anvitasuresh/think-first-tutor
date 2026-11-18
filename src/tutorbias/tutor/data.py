"""Prompts for the tutor: GRPO training prompts (stage 4) and evaluation prompts (stage 5).

  python -m tutorbias.tutor.data --config configs/grpo.yaml     # -> data/processed/grpo_train.jsonl

Training prompts. Each rl_train question appears once, with its stuck-student message in ONE dialect
group, assigned round-robin so every group gets the same number of questions. This is what lets bias
propagate: if the reward judge scores hints lower for AAE students, the policy sees that signal on a
quarter of its training prompts. A variant the transform left unchanged (or that changed a number) is
replaced by the next group that has a usable variant, and the swap is counted in the stats.

Evaluation prompts. Every test question with every version of its message (dialect groups, the
paraphrase control, and names), so per-group gaps in tutor behaviour are paired, like the audit.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from ..io import load_config, read_jsonl, write_jsonl
from ..replies.generate import tutor_conv
from ..textutil import garbled_reason


def usable(v: dict[str, Any]) -> bool:
    """A variant is usable if it keeps every number, isn't garbled by a known transform bug, and (for a
    non-MAE dialect group) actually differs."""
    return (bool(v.get("numbers_ok", True)) and garbled_reason(v.get("text", "")) is None
            and (v["group"] == "mae" or bool(v.get("changed", True))))


def prompt_row(item: dict[str, Any], v: dict[str, Any], id_prefix: str) -> dict[str, Any]:
    return {
        "prompt_id": f"{id_prefix}::{v['variant_id']}",
        "item_id": item["item_id"], "variant_id": v["variant_id"], "message_type": v["message_type"],
        "axis": v["axis"], "group": v["group"], "name": v.get("name"), "name_group": v.get("name_group"),
        "changed": v.get("changed", False),
        "question": item["question"], "answer": item["final_answer"],
        "student_message": v["text"], "base_message": v["base_text"],
        # Exactly what the tutor sees: no style instruction (the policy has to learn its own style).
        "prompt": tutor_conv(item["question"], v["text"], style=None),
    }


def build_train_prompts(items: list[dict], variants: list[dict], groups: list[str], seed: int,
                        limit: int | None = None) -> tuple[list[dict], dict]:
    by_item: dict[str, dict[str, dict]] = {}
    for v in variants:
        if v["axis"] == "dialect" and v["group"] in groups and usable(v):
            by_item.setdefault(v["item_id"], {})[v["group"]] = v
    rng = random.Random(seed)
    items = [it for it in items if it["item_id"] in by_item]
    rng.shuffle(items)
    if limit:
        items = items[:limit]
    rows, swaps = [], Counter()
    for k, it in enumerate(items):
        want = groups[k % len(groups)]
        avail = by_item[it["item_id"]]
        # Round-robin target; if that group's variant isn't usable, take the next group in order.
        order = groups[k % len(groups):] + groups[: k % len(groups)]
        got = next(g for g in order if g in avail)
        if got != want:
            swaps[f"{want}->{got}"] += 1
        rows.append(prompt_row(it, avail[got], "train"))
    stats = {"n_prompts": len(rows), "by_group": dict(Counter(r["group"] for r in rows)),
             "swapped_unusable_variant": dict(swaps)}
    return rows, stats


def build_eval_prompts(items: list[dict], variants: list[dict], groups: list[str] | None = None) -> list[dict]:
    """All usable versions of each test question's message (None = every group, names included)."""
    by_id = {it["item_id"]: it for it in items}
    rows = []
    for v in variants:
        if v["item_id"] in by_id and usable(v) and (groups is None or v["group"] in groups):
            rows.append(prompt_row(by_id[v["item_id"]], v, "eval"))
    rows.sort(key=lambda r: (r["item_id"], r["group"], r["name"] or ""))
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/grpo.yaml")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    d = cfg["data"]
    items = read_jsonl(d["items"])
    variants = [v for v in read_jsonl(d["variants"]) if v["split"] == "rl_train"]
    rows, stats = build_train_prompts(items, variants, d["groups"], cfg["seed"], d.get("limit"))
    write_jsonl(d["train_prompts"], rows)
    Path(d["train_prompts"]).with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
