"""Build the interactive demo page from the project's saved outputs.

  python demo/build.py            # -> demo/site/index.html (one self-contained file, no server or GPU)

The page is for visitors, not for the full results (those are in the README). Its parts:

  1. "Try to get the answer."  A chat where you pick what to say and the trained tutor replies. Every
     reply is a real one the tutor wrote during evaluation (results/eval/replies.jsonl, Mainstream US
     messages, replies cut off by the length limit skipped, sampled at random). A toggle shows what the
     untrained base model said to the same message. It is not a live model.
  2. "How the tutor learned."  GRPO in plain words, with the real training curve
     (results/eval/training_curve.csv) and the before/after numbers (results/eval/metrics_by_group.csv).
  3. "Hint or answer?"  A guessing game on the 40 replies Anvita labeled by hand (E1). Uses
     data/labels/*, results/judge/label_set_scores.jsonl and results/calibration/all/metrics.json.
  4. "Same help, different student."  One tutor reply shown next to every version of the student's
     message, with the grader's score for each. Variants come from data/processed/variants.jsonl when
     present, else demo/variants_sample.jsonl, a committed copy of just the demo's rows.
  5. "What we found."  Four plain takeaways, linking to the README for the numbers.

Only the standard library (plus the repo's own src/tutorbias/textutil.py) is used, so this runs anywhere
with Python 3.10+, no install needed. Rerun it after each Kaggle
stage finishes and commit demo/site/index.html; the GitHub Pages workflow publishes it.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Same list of known Multi-VALUE breakages ("what I ai wrong", "I am did with it") that the audit
# drops, so the demo never shows broken text labeled as a dialect. textutil only needs the stdlib.
from tutorbias.textutil import garbled_reason  # noqa: E402

DEMO = ROOT / "demo"
OUT = DEMO / "site" / "index.html"

# How each student group is shown on the page. Order is the order of the tabs.
GROUPS = {
    "mae": {"label": "Mainstream American English", "short": "Mainstream US", "axis": "dialect"},
    "aae": {"label": "African American English", "short": "AAE", "axis": "dialect"},
    "indian": {"label": "Indian English", "short": "Indian English", "axis": "dialect"},
    "l2": {"label": "English learner", "short": "Learner", "axis": "dialect"},
    "paraphrase": {"label": "Reworded, no dialect (control)", "short": "Reworded", "axis": "control"},
    "name_white": {"label": "Name often read as white", "short": "White-coded name", "axis": "name"},
    "name_black": {"label": "Name often read as Black", "short": "Black-coded name", "axis": "name"},
    "name_hispanic": {"label": "Name often read as Hispanic", "short": "Hispanic-coded name", "axis": "name"},
    "name_south_asian": {"label": "Name often read as South Asian", "short": "South Asian-coded name", "axis": "name"},
}
LABEL_WORDS = {-1: "Gives it away", 0: "Neither", 1: "Guides"}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def human_labels() -> dict[str, dict]:
    """Anvita's spot-check labels; if she relabeled an item, the last label wins."""
    out = {}
    with open(ROOT / "data/labels/labels_anvita.csv") as f:
        for r in csv.DictReader(f):
            out[r["label_id"]] = {"scaffolding": int(r["scaffolding"]), "helpfulness": int(r["helpfulness"])}
    return out


# ------------------------------------------------------------------------------------------ part 1
def build_game() -> dict:
    items = {r["label_id"]: r for r in read_jsonl(ROOT / "data/labels/label_set.jsonl")}
    key = {r["label_id"]: r for r in read_jsonl(ROOT / "data/labels/label_key.jsonl")}
    judge = {r["label_id"]: r for r in read_jsonl(ROOT / "results/judge/label_set_scores.jsonl")}
    human = human_labels()
    rounds = []
    for lid in (ROOT / "data/labels/spotcheck_ids.txt").read_text().split():
        if lid not in human:
            continue
        it, j = items[lid], judge[lid]
        rounds.append({
            "id": lid, "question": it["question"], "answer": it["answer"],
            "student": it["student_message"], "reply": it["reply"], "style": key[lid]["style"],
            "item_id": key[lid]["item_id"],
            "human": human[lid]["scaffolding"], "human_help": human[lid]["helpfulness"],
            "judge": round(j["scaffolding"], 3), "judge_label": j["scaffolding_label"],
            "judge_help": round(j["helpfulness"], 2), "rationale": j.get("rationale"),
        })
    # Fixed shuffle so the game doesn't open on five replies of the same kind.
    random.Random(7).shuffle(rounds)
    metrics = json.loads((ROOT / "results/calibration/all/metrics.json").read_text())["judge_vs_human"]
    calib = {"n": metrics["n"], "kappa": metrics["scaffolding_kappa"], "kappa_ci": metrics["scaffolding_kappa_ci"],
             "accuracy": metrics["scaffolding_accuracy"], "spearman": metrics["helpfulness_spearman"]}
    return {"rounds": rounds, "calibration": calib}


# ------------------------------------------------------------------------------------------ part 2
def load_variants(messages: set[tuple[str, str]]) -> tuple[dict[str, dict[str, dict]], str]:
    """(item_id, MAE message text) -> group -> variant, for the demo's messages.

    Student messages come from templates, so the same text appears under many questions; matching on
    the item id as well keeps exactly the variants the audit built for each demo question.

    The audit's variants file (data/processed/variants.jsonl, made by `run_stage.sh variants`) is large
    and gitignored, so when it is present we copy just the demo's rows into demo/variants_sample.jsonl,
    which is committed. The page therefore always shows the audit's real Multi-VALUE variants, never
    hand-written dialect.
    """
    full = ROOT / "data/processed/variants.jsonl"
    sample = DEMO / "variants_sample.jsonl"
    if full.exists():
        rows = read_jsonl(full)
        base = {r["message_id"]: r["text"] for r in rows if r["group"] == "mae"}
        cols = ("variant_id", "item_id", "group", "name", "text", "changed", "features", "numbers_ok")
        keep = [{"source_text": base[r["message_id"]], **{c: r.get(c) for c in cols}} for r in rows
                if (r["item_id"], base.get(r["message_id"])) in messages and r["group"] in GROUPS]
        keep.sort(key=lambda r: (r["source_text"], r["group"], r.get("variant_id", "")))
        sample.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    by_msg: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    if not sample.exists():
        return by_msg, "none"
    for r in read_jsonl(sample):
        by_msg[(r["item_id"], r["source_text"])].setdefault(r["group"], r)  # first name in each name group
    return by_msg, "audit variants (Multi-VALUE, configs/data.yaml)"


def load_pair_scores() -> dict[tuple[str, str], dict[str, float]]:
    """(reply text, student message) -> scorer -> score, from the audit's per-pair score files."""
    pairs_path = ROOT / "data/processed/audit_pairs.jsonl"
    sdir = ROOT / "results/audit/scores"
    files = {"judge_scaffolding": ("judge.jsonl", "scaffolding"), "skywork_v2": ("skywork_v2.jsonl", "reward"),
             "tulu3_rm": ("tulu3_rm.jsonl", "reward")}
    if not pairs_path.exists() or not sdir.exists():
        return {}
    pairs = {p["pair_id"]: p for p in read_jsonl(pairs_path)}
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for scorer, (fname, col) in files.items():
        if (sdir / fname).exists():
            for r in read_jsonl(sdir / fname):
                p = pairs.get(r["id"])
                if p is not None and r.get(col) is not None:
                    out[(p["reply"], p["student_message"])][scorer] = r[col]
    return out


def build_voices(game: dict) -> dict:
    by_msg, source = load_variants({(r["item_id"], r["student"]) for r in game["rounds"]})
    pair_scores = load_pair_scores()
    examples = []
    # Showcase good help: replies Anvita labeled "Guides", so the question is whether every student
    # gets the same credit for the same good hint. Prefer messages where all three dialects changed.
    for r in game["rounds"]:
        v = by_msg.get((r["item_id"], r["student"]))
        if r["human"] != 1 or not v:
            continue
        variants = {"mae": {"text": r["student"], "features": {}}}
        for g in GROUPS:
            if (g != "mae" and g in v and v[g].get("changed", True) and v[g].get("numbers_ok", True)
                    and not garbled_reason(v[g]["text"])):
                variants[g] = {"text": v[g]["text"], "features": v[g].get("features", {}), "name": v[g].get("name")}
        n_dialect = sum(1 for g in ("aae", "indian", "l2") if g in variants)
        if n_dialect < 2:
            continue
        scores = {g: pair_scores[(r["reply"], x["text"])] for g, x in variants.items()
                  if (r["reply"], x["text"]) in pair_scores}
        examples.append({"id": r["id"], "question": r["question"], "reply": r["reply"], "style": r["style"],
                         "variants": variants, "scores": scores, "_rank": -n_dialect})
    # Most dialects first; among those, open on a typical example (smallest judge change across versions),
    # since the audit-wide gaps are small. Bigger swings come later and the page says one example can vary.
    def spread(e: dict) -> float:
        js = [x.get("judge_scaffolding") for x in e["scores"].values() if x.get("judge_scaffolding") is not None]
        return max(js) - min(js) if js else 0.0

    examples.sort(key=lambda e: (e["_rank"], spread(e)))
    for e in examples:
        del e["_rank"]
    return {"examples": examples[:8], "variant_source": source}


# ------------------------------------------------------------------------------------------ chat + training
# What the visitor can "say" in the chat, mapped to the eval's four kinds of student message.
CHAT_MOVES = {
    "wants_answer": "Just tell me the answer",
    "dont_get": "I don't get it",
    "wrong_answer": "I got it wrong and don't know why",
    "tried_partial": "I started but I'm stuck",
}


def finished(reply: str) -> bool:
    """False for replies cut off by the 256-token limit (they end mid-sentence)."""
    return reply.rstrip()[-1:] in ".?!)*"


def build_chat(per_move: int = 24) -> dict:
    """Real recorded replies for the chat: the trained tutor and the untrained base model, same prompt.

    Only Mainstream US messages, only replies that were not cut off, sampled at random (fixed seed), so
    the share of replies that leak the answer is roughly the real one, not cherry-picked.
    """
    replies_path = ROOT / "results/eval/replies.jsonl"
    judge_path = ROOT / "results/eval/scores/judge.jsonl"
    if not replies_path.exists():
        return {"moves": CHAT_MOVES, "pool": {}}
    judge = {r["id"]: r for r in read_jsonl(judge_path)} if judge_path.exists() else {}
    by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in read_jsonl(replies_path):
        if r.get("group") == "mae":
            j = judge.get(r["eval_id"], {})
            by_prompt[r["prompt_id"]][r["model"]] = {**r, "judge_label": j.get("scaffolding_label")}
    pool: dict[str, list[dict]] = defaultdict(list)
    for models in by_prompt.values():
        b, t = models.get("base"), models.get("tutor_a")
        if not b or not t or not finished(t["reply"]):
            continue
        pool[t["message_type"]].append({
            "question": t["question"], "answer": t["answer"], "student": t["student_message"],
            "tutor": t["reply"], "tutor_leak": bool(t["answer_leaked"]), "tutor_label": t["judge_label"],
            "base": b["reply"], "base_leak": bool(b["answer_leaked"]), "base_label": b["judge_label"],
            "base_cut": not finished(b["reply"])})
    rng = random.Random(11)
    out = {}
    for move in CHAT_MOVES:
        items = sorted(pool.get(move, []), key=lambda e: e["question"])
        rng.shuffle(items)
        out[move] = items[:per_move]
    return {"moves": CHAT_MOVES, "pool": out}


def build_training() -> dict:
    """The real GRPO training curve (10-step bins of logged rewards) and the headline before/after numbers."""
    out: dict = {"curve": [], "headline": {}}
    curve = ROOT / "results/eval/training_curve.csv"
    if curve.exists():
        with open(curve) as f:
            out["curve"] = [{"step": float(r["bin"]), "guides": float(r["scaffolding"]),
                             "leak": float(r["answer_leaked"])} for r in csv.DictReader(f)]
    replies, judge = ROOT / "results/eval/replies.jsonl", ROOT / "results/eval/scores/judge.jsonl"
    if replies.exists() and judge.exists():
        # Pooled over every test reply, the same way as the README's E5 numbers and the poster chart
        # (scripts/plot_tutor.py): "hint" = judge score above 0, "leak" = final answer appears.
        score = {r["id"]: r for r in read_jsonl(judge)}
        acc: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for r in read_jsonl(replies):
            j = score.get(r["eval_id"])
            if j:
                acc[r["model"]].append((j["scaffolding"] > 0, bool(r["answer_leaked"]), j["helpfulness"]))
        out["headline"] = {m: {"hint": sum(x[0] for x in v) / len(v), "leak": sum(x[1] for x in v) / len(v),
                               "help": sum(x[2] for x in v) / len(v)} for m, v in acc.items()}
    return out


# ------------------------------------------------------------------------------------------ page
def main() -> None:
    game = build_game()
    data = {"game": game, "voices": build_voices(game), "chat": build_chat(), "training": build_training(),
            "groups": GROUPS, "label_words": LABEL_WORDS}
    body = (DEMO / "template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    body = body.replace("/*__DATA__*/null", payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # template.html is page content only; wrap it in a full document for GitHub Pages.
    page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    OUT.write_text(page)
    v = data["voices"]
    print(f"[demo] game rounds: {len(game['rounds'])} | voice examples: {len(v['examples'])} "
          f"(variants: {v['variant_source']}) | chat replies: "
          f"{ {k: len(x) for k, x in data['chat']['pool'].items()} } | wrote {OUT}")


if __name__ == "__main__":
    main()
