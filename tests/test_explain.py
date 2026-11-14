"""Stage 3 analyses on synthetic data with known structure."""

import numpy as np
import pandas as pd

from tutorbias.explain.features import feature_effects, pair_deltas
from tutorbias.explain.probes import alignment, grouped_cv_accuracy
from tutorbias.explain.rationales import analyze, mentions_language, pick_subset


def synthetic_pairs(n_items=150, seed=0):
    """Feature 'copula_deletion' lowers the score by 0.5; 'negative_concord' does nothing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        for k in range(3):
            base = rng.normal()
            rid = f"q{i}r{k}"
            rows.append({"reply_id": rid, "item_id": f"q{i}", "axis": "dialect", "group": "mae", "changed": False,
                         "features": {}, "s": base})
            for g in ("aae", "l2"):
                f = {}
                if rng.random() < 0.5:
                    f["copula_deletion"] = 1
                if rng.random() < 0.5:
                    f["negative_concord"] = 1
                s = base - 0.5 * ("copula_deletion" in f) + rng.normal(0, 0.05)
                rows.append({"reply_id": rid, "item_id": f"q{i}", "axis": "dialect", "group": g, "changed": bool(f),
                             "features": f, "s": s})
    return pd.DataFrame(rows)


def test_feature_effects_recover_injected_feature():
    v = pair_deltas(synthetic_pairs(), "s")
    eff = feature_effects(v, min_count=10, n_boot=200, n_perm=100, seed=0).set_index("feature")
    sd = float(np.std(synthetic_pairs().query("group == 'mae'")["s"]))
    assert eff.loc["copula_deletion", "beta"] * sd < -0.4
    assert eff.loc["copula_deletion", "beyond_null"]
    assert not eff.loc["negative_concord", "beyond_null"]


def test_mentions_language():
    assert mentions_language("The student's grammar is informal, but the hint is good.")
    assert mentions_language("The reply ignores the dialect of the message.")
    assert not mentions_language("The tutor gives away the final answer of 72.")
    assert not mentions_language(None)


def test_rationale_analysis_bins():
    rows = pd.DataFrame([
        {"pair_id": "a|mae", "reply_id": "a", "group": "mae", "scaffolding": 0.8, "rationale": "Good hint."},
        {"pair_id": "a|aae", "reply_id": "a", "group": "aae", "scaffolding": 0.2, "rationale": "Good hint."},
        {"pair_id": "b|mae", "reply_id": "b", "group": "mae", "scaffolding": 0.5, "rationale": "Asks a question."},
        {"pair_id": "b|aae", "reply_id": "b", "group": "aae", "scaffolding": 0.5, "rationale": "Asks a question."},
    ])
    t, v = analyze(rows, large_gap=0.2, no_gap=0.02)
    assert set(v["gap_bin"]) == {"large", "none"}
    large = t[t["gap_bin"] == "large"].iloc[0]
    assert large["mention_rate"] == 0 and large["reason_drift"] == 0


def test_pick_subset_balances_styles():
    pairs = []
    for r in range(40):
        style = ["hint", "answer"][r % 2]
        for g in ("mae", "aae", "paraphrase"):
            pairs.append({"reply_id": f"r{r}", "item_id": f"q{r}", "style": style, "axis": "control" if g == "paraphrase" else "dialect",
                          "group": g, "pair_id": f"r{r}|{g}"})
    sub = pick_subset(pairs, 10, ["mae", "aae", "paraphrase"], seed=0)
    replies = {p["reply_id"]: p["style"] for p in sub}
    assert len(replies) == 10 and list(replies.values()).count("hint") == 5
    assert len(sub) == 30


def test_probe_and_alignment_on_planted_direction():
    rng = np.random.default_rng(0)
    n, d = 120, 16
    score_dir = np.zeros(d)
    score_dir[0] = 1.0
    meta, xs = [], []
    for i in range(n):
        base = rng.normal(size=d)
        for g, shift in (("mae", 0.0), ("aae", -3.0)):
            x = base + shift * score_dir + rng.normal(0, 0.1, size=d)
            xs.append(x)
            meta.append({"reply_id": f"r{i}", "item_id": f"q{i // 2}", "group": g, "scaffolding": float(x[0])})
    x = np.array(xs)
    meta = pd.DataFrame(meta)
    y = (meta["group"] == "aae").to_numpy(int)
    assert grouped_cv_accuracy(x, y, meta["item_id"].to_numpy(), 5, 0) > 0.85
    al = alignment(x, meta, "aae")
    assert al["cosine"] < -0.8 and al["spearman_shift_vs_gap"] > 0.5
