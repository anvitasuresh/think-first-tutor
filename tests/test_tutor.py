"""Stage 4-5 logic that doesn't need a GPU: prompts, reward, config handling, E5-E7 analysis."""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from tutorbias.io import read_jsonl
from tutorbias.judge.judge import ScaffoldingJudge
from tutorbias.models.backends import FakeBackend
from tutorbias.tutor.analyze import did, gaps, generic_question, vs_base
from tutorbias.tutor.data import build_eval_prompts, build_train_prompts
from tutorbias.tutor.grpo import assign_gpus, filter_kwargs, last_checkpoint, run_config
from tutorbias.tutor.reward import LocalJudge, combine, completion_text, judge_rows, make_reward_func

GROUPS = ["mae", "aae", "indian", "l2"]


def make_variants(n_items, unchanged=()):
    items, variants = [], []
    for i in range(n_items):
        iid = f"q{i}"
        items.append({"item_id": iid, "question": f"Q{i}: 3 + 4?", "final_answer": "7"})
        for g in GROUPS + ["paraphrase", "name_black"]:
            axis = "dialect" if g in GROUPS else ("control" if g == "paraphrase" else "name")
            variants.append({"variant_id": f"{iid}::{g}", "item_id": iid, "split": "rl_train", "message_type": "dont_get",
                             "axis": axis, "group": g, "text": f"{g} stuck", "base_text": "mae stuck",
                             "changed": g != "mae" and (iid, g) not in unchanged, "numbers_ok": True})
    return items, variants


def test_train_prompts_balanced_and_swap_unusable():
    items, variants = make_variants(40, unchanged={("q0", "aae"), ("q1", "aae"), ("q2", "aae"), ("q3", "aae")})
    rows, stats = build_train_prompts(items, variants, GROUPS, seed=0)
    assert len(rows) == 40 and len({r["item_id"] for r in rows}) == 40
    assert all(r["axis"] == "dialect" for r in rows)
    counts = stats["by_group"]
    assert max(counts.values()) - min(counts.values()) <= 2
    assert all(r["student_message"] == f"{r['group']} stuck" for r in rows)
    assert rows[0]["prompt"][1]["content"].endswith(rows[0]["student_message"])
    # no unchanged AAE variant was used
    assert not any(r["group"] == "aae" and r["item_id"] in {"q0", "q1", "q2", "q3"} for r in rows)


def test_eval_prompts_cover_every_group():
    items, variants = make_variants(3)
    rows = build_eval_prompts(items, variants)
    assert len(rows) == 3 * 6
    assert {r["group"] for r in rows} == set(GROUPS) | {"paraphrase", "name_black"}


def fake_judge():
    # Scores hints (a "?" in the reply) high and answers low; penalizes nothing about the student.
    def lp(conv, prefix, cands):
        user = conv[1]["content"]
        hint = user.rstrip().endswith("?")
        if prefix.endswith("SCAFFOLDING: "):
            p = {"0": 0.05, "1": 0.15, "2": 0.8} if hint else {"0": 0.8, "1": 0.15, "2": 0.05}
        else:
            p = {"1": 0.1, "2": 0.1, "3": 0.2, "4": 0.4, "5": 0.2}
        return {k: math.log(v) for k, v in p.items()}

    return ScaffoldingJudge(FakeBackend(lp), "v1")


def test_reward_func_scores_and_logs(tmp_path):
    items, variants = make_variants(2)
    rows, _ = build_train_prompts(items, variants, GROUPS, seed=0)
    cols = {k: [r[k] for r in rows] for k in ("item_id", "group", "question", "answer", "student_message", "base_message")}
    comps = [[{"role": "assistant", "content": "What is 3 + 4?"}], [{"role": "assistant", "content": "It is 7."}]]
    f = make_reward_func(LocalJudge(fake_judge()), {"w_scaffolding": 1.0, "w_helpfulness": 0.5}, tmp_path / "r.jsonl")
    rewards = f(prompts=[r["prompt"] for r in rows], completions=comps, **cols)
    assert rewards[0] > 0.5 > rewards[1]
    log = read_jsonl(tmp_path / "r.jsonl")
    assert len(log) == 2 and log[1]["answer_leaked"] and not log[0]["answer_leaked"]
    assert {"group", "scaffolding", "helpfulness", "reward"} <= set(log[0])


def test_judge_sees_switch():
    cols = {"question": ["Q"], "answer": ["7"], "student_message": ["aae text"], "base_message": ["mae text"]}
    assert judge_rows(["hi"], cols, "student")[0]["student_message"] == "aae text"
    assert judge_rows(["hi"], cols, "mae")[0]["student_message"] == "mae text"
    with pytest.raises(ValueError):
        judge_rows(["hi"], cols, "other")
    assert completion_text("x") == "x" and completion_text([{"role": "assistant", "content": "y"}]) == "y"
    assert combine(1.0, 5.0, 1.0, 0.5) == pytest.approx(1.5)
    assert combine(-1.0, 1.0, 1.0, 0.5) == pytest.approx(-1.5)


def test_config_helpers(tmp_path):
    @dataclasses.dataclass
    class Conf:
        learning_rate: float = 1e-5
        beta: float = 0.0

    kept, dropped = filter_kwargs(Conf, {"learning_rate": 1.0, "max_prompt_length": 3})
    assert kept == {"learning_rate": 1.0} and dropped == ["max_prompt_length"]
    assert assign_gpus(2) == ("0", "1") and assign_gpus(1) == ("0", "0") and assign_gpus(0) == (None, None)
    for step, complete in ((5, True), (10, True), (15, False)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if complete:
            (d / "trainer_state.json").write_text("{}")
    assert last_checkpoint(tmp_path).endswith("checkpoint-10")
    cfg = {"seed": 1, "reward": {"judge_sees": "student", "w_scaffolding": 1.0}, "runs": {"b": {"reward": {"judge_sees": "mae"}}}}
    rc = run_config(cfg, "b")
    assert rc["reward"] == {"judge_sees": "mae", "w_scaffolding": 1.0} and "runs" not in rc


def synthetic_eval(n_items=120, seed=0):
    """Base model: no gap. Tutor: scaffolds more overall, but less for AAE students (propagation)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        for model in ("base", "tutor_a"):
            for g in ("mae", "aae", "indian"):
                s = rng.normal(0.0, 0.3) + (0.4 if model == "tutor_a" else 0) - (0.3 if (model == "tutor_a" and g == "aae") else 0)
                rows.append({"model": model, "item_id": f"q{i}", "group": g, "prompt_id": f"q{i}::{g}",
                             "scaffolding": s, "answer_leaked": float(s < -0.2)})
    return pd.DataFrame(rows)


def test_e5_e7_detect_injected_propagation():
    df = synthetic_eval()
    kw = {"n_boot": 300, "seed": 0, "alpha": 0.05}
    e5 = vs_base(df, "base", **kw)
    sc = e5[e5["metric"] == "scaffolding"].iloc[0]
    assert sc["diff"] > 0.2 and sc["ci_lo"] > 0
    d = did(df, ["scaffolding"], "base", "mae", **kw).set_index("group")
    assert d.loc["aae", "did"] < -0.15 and d.loc["aae", "ci_hi"] < 0
    assert d.loc["indian", "ci_lo"] < 0 < d.loc["indian", "ci_hi"]
    g = gaps(df, ["scaffolding"], "mae", **kw)
    assert set(g["model"]) == {"base", "tutor_a"}


def test_generic_question():
    assert generic_question("What do you think?", "Tom has 3 apples and 4 pears.")
    assert not generic_question("What is 3 + 4?", "Tom has 3 apples and 4 pears.")
    assert not generic_question("You have 7 fruits.", "Tom has 3 apples and 4 pears.")
