"""Bias regression test: the graders that train the tutor must not penalize any student group.

The first test runs on the committed audit results (results/audit/gaps.csv), so a rerun that makes the
training grader biased fails CI. The others check the gate and the power calculation on toy numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tutorbias.audit.gate import checked, main, violations
from tutorbias.audit.power import power_table
from tutorbias.io import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs/bias_gate.yaml")
GAPS = ROOT / CFG["gaps_file"]


@pytest.mark.skipif(not GAPS.exists(), reason="audit results not committed")
def test_training_graders_pass_bias_gate():
    gaps = pd.read_csv(GAPS)
    assert checked(gaps, CFG, CFG["gated_scorers"]) > 0, "gate found nothing to check"
    v = violations(gaps, CFG, CFG["gated_scorers"])
    assert v.empty, f"training grader penalizes a student group:\n{v.to_string()}"


@pytest.mark.skipif(not GAPS.exists(), reason="audit results not committed")
def test_gate_catches_known_off_the_shelf_penalty():
    # Tulu-3-RM's Indian English penalty is the one known violation. If this starts failing, the audit
    # results changed and the README / poster numbers need rechecking.
    v = violations(pd.read_csv(GAPS), CFG, CFG["report_only"])
    assert ((v["scorer"] == "tulu3_rm") & (v["group"] == "indian")).any()


def toy_gaps(ci_lo_aae: float) -> pd.DataFrame:
    return pd.DataFrame([
        {"subset": "all", "scorer": "judge_scaffolding", "group": "aae", "vs": "mae", "gap_std": 0.0,
         "ci_lo": ci_lo_aae, "ci_hi": 0.02, "net_of_paraphrase_std": 0.0, "net_ci_lo": -0.01, "net_ci_hi": 0.01,
         "n_items": 300},
        {"subset": "all", "scorer": "judge_scaffolding", "group": "name_black", "vs": "name_white", "gap_std": 0.0,
         "ci_lo": -0.01, "ci_hi": 0.01, "net_of_paraphrase_std": np.nan, "net_ci_lo": np.nan, "net_ci_hi": np.nan,
         "n_items": 300},
    ])


def test_gate_flags_penalty_beyond_limit(tmp_path):
    assert violations(toy_gaps(-0.02), CFG, ["judge_scaffolding"]).empty
    v = violations(toy_gaps(-0.20), CFG, ["judge_scaffolding"])
    assert list(v["group"]) == ["aae"] and list(v["column"]) == ["ci_lo"]
    f = tmp_path / "gaps.csv"
    toy_gaps(-0.20).to_csv(f, index=False)
    assert main(["--config", str(ROOT / "configs/bias_gate.yaml"), "--gaps", str(f)]) == 1


def test_gate_fails_when_nothing_is_checked(tmp_path):
    f = tmp_path / "gaps.csv"
    toy_gaps(-0.02).assign(scorer="other").to_csv(f, index=False)
    assert main(["--config", str(ROOT / "configs/bias_gate.yaml"), "--gaps", str(f)]) == 1


def test_power_mde_matches_formula():
    pt = power_table(toy_gaps(-0.02), alpha=0.05, power=0.8)
    se = 0.04 / (2 * 1.959964)
    # 2 tests in the family -> Bonferroni z = z(1 - 0.05/4) = 2.2414; z(0.8) = 0.8416
    assert pt.loc[0, "mde"] == pytest.approx((2.2414 + 0.8416) * se, rel=1e-3)
    assert pt.loc[0, "largest_penalty"] == -0.02
