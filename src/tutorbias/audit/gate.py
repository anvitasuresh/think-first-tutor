"""Bias regression gate: fail if a grader penalizes a student group by more than a set limit.

  python -m tutorbias.audit.gate --config configs/bias_gate.yaml

Reads the audit's gaps.csv and checks, for each scorer, group and CI column listed in the config, that
the pessimistic end of the 95% CI is not below -max_penalty_sd. Exit code 1 if a gated scorer (one that
trains the tutor) fails; off-the-shelf RMs are listed as report-only. tests/test_bias_gate.py runs the same
check in CI, so a change that makes the training grader biased breaks the build.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd

from ..io import load_config


def violations(gaps: pd.DataFrame, cfg: dict[str, Any], scorers: list[str]) -> pd.DataFrame:
    """Rows (scorer, group, vs, column, value) where the CI bound is below -max_penalty_sd."""
    limit = -float(cfg["max_penalty_sd"])
    g = gaps[(gaps["subset"] == cfg.get("subset", "all")) & gaps["scorer"].isin(scorers)]
    out = []
    for chk in cfg["checks"]:
        sub = g[(g["vs"] == chk["vs"]) & g["group"].isin(chk["groups"])]
        for col in chk["columns"]:
            if col not in sub:
                continue
            bad = sub[sub[col].notna() & (sub[col] < limit)]
            for _, r in bad.iterrows():
                out.append({"scorer": r["scorer"], "group": r["group"], "vs": r["vs"], "column": col,
                            "value": float(r[col]), "limit": limit})
    return pd.DataFrame(out, columns=["scorer", "group", "vs", "column", "value", "limit"])


def checked(gaps: pd.DataFrame, cfg: dict[str, Any], scorers: list[str]) -> int:
    """How many (scorer, group, vs) rows the config covers, so an empty gaps file cannot pass silently."""
    g = gaps[(gaps["subset"] == cfg.get("subset", "all")) & gaps["scorer"].isin(scorers)]
    return sum(int(((g["vs"] == c["vs"]) & g["group"].isin(c["groups"])).sum()) for c in cfg["checks"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/bias_gate.yaml")
    p.add_argument("--gaps", default=None, help="override gaps_file from the config")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    gaps = pd.read_csv(a.gaps or cfg["gaps_file"])
    rc = 0
    for kind, scorers in (("GATED", cfg["gated_scorers"]), ("report only", cfg.get("report_only", []))):
        n, v = checked(gaps, cfg, scorers), violations(gaps, cfg, scorers)
        print(f"[{kind}] {', '.join(scorers)}: {n} gaps checked, {len(v)} over the limit "
              f"of {cfg['max_penalty_sd']} SD")
        for _, r in v.iterrows():
            print(f"   {r['scorer']:<26} {r['group']:<18} vs {r['vs']:<11} {r['column']:<10} {r['value']:+.3f}")
        if kind == "GATED" and (len(v) or n == 0):
            rc = 1
    print("PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
