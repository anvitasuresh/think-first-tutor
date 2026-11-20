"""Power check for E2: how small a gap could the audit have detected?

  python -m tutorbias.audit.power --gaps results/audit/gaps.csv --out results/audit

Several audit results are "no penalty". That claim is only as strong as the audit's resolution: with too
few questions, a real gap would also come out "not significant". This script reports two numbers per
(scorer, group) test, both in SD units, read straight off gaps.csv:

  * MDE (minimum detectable effect): the smallest true gap the audit would detect with 80% power at the
    family-wise alpha used for the results. SE is taken from the cluster-bootstrap 95% CI
    (SE = CI width / (2 * 1.96)). We use Bonferroni (alpha / m tests) as a conservative stand-in for
    Holm, so the real MDE is at most this large.
        MDE = (z[1 - alpha / (2m)] + z[power]) * SE
  * Largest plausible penalty: how negative the gap could still be, given the data (the lower end of the
    95% CI, or 0 if the CI lies above 0). "No penalty larger than X SD" is the honest version of
    "no penalty".

The same is done for the gap net of the paraphrase control when it exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

Z95 = norm.ppf(0.975)


def se_from_ci(lo: pd.Series, hi: pd.Series) -> pd.Series:
    """Standard error implied by a 95% CI (normal approximation)."""
    return (hi - lo) / (2 * Z95)


def power_table(gaps: pd.DataFrame, alpha: float = 0.05, power: float = 0.8) -> pd.DataFrame:
    """One row per (subset, scorer, group, vs) with the MDE and the largest penalty the CI still allows."""
    rows = []
    for subset, g in gaps.groupby("subset", sort=False):
        m = len(g)  # tests in the Holm family for this subset
        z = norm.ppf(1 - alpha / (2 * m)) + norm.ppf(power)
        for _, r in g.iterrows():
            row = {"subset": subset, "scorer": r["scorer"], "group": r["group"], "vs": r["vs"],
                   "n_items": r["n_items"], "gap_std": r["gap_std"],
                   "se": se_from_ci(r["ci_lo"], r["ci_hi"]),
                   "largest_penalty": min(0.0, r["ci_lo"])}
            row["mde"] = z * row["se"]
            if pd.notna(r.get("net_of_paraphrase_std")):
                se_net = se_from_ci(r["net_ci_lo"], r["net_ci_hi"])
                row |= {"net_std": r["net_of_paraphrase_std"], "net_mde": z * se_net,
                        "net_largest_penalty": min(0.0, r["net_ci_lo"])}
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(pt: pd.DataFrame, path: Path) -> None:
    t = pt[pt["subset"] == "all"]
    lines = ["# E2 power check", "",
             "How small a gap could the audit have caught? All numbers are in SD units of each scorer's MAE "
             "scores. **MDE** = smallest true gap detected with 80% power at the family-wise alpha "
             "(Bonferroni over all tests, so conservative). **Largest penalty** = the most negative gap still "
             "inside the 95% CI (0 if the CI is above 0). For scale: 0.2 SD is a conventional \"small\" effect.",
             "", "| Scorer | Group | vs | Gap | MDE | Largest penalty | Net gap | Net MDE | Net largest penalty |",
             "|---|---|---|---|---|---|---|---|---|"]

    def f(x: float) -> str:
        return "" if pd.isna(x) else f"{x:+.3f}"

    for _, r in t.iterrows():
        lines.append(f"| {r['scorer']} | {r['group']} | {r['vs']} | {f(r['gap_std'])} | {r['mde']:.3f} "
                     f"| {f(r['largest_penalty'])} | {f(r.get('net_std'))} | "
                     f"{'' if pd.isna(r.get('net_mde')) else format(r['net_mde'], '.3f')} "
                     f"| {f(r.get('net_largest_penalty'))} |")
    dial = t[t["vs"] == "mae"]
    lines += ["", "## In one line", "",
              f"- Raw gaps: MDE {dial['mde'].min():.3f} to {dial['mde'].max():.3f} SD "
              f"(median {dial['mde'].median():.3f}).",
              f"- Net of paraphrase: MDE {np.nanmin(dial.get('net_mde', np.nan)):.3f} to "
              f"{np.nanmax(dial.get('net_mde', np.nan)):.3f} SD. The paraphrase control adds noise, so net "
              "effects are measured less precisely than raw gaps."]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gaps", default="results/audit/gaps.csv")
    p.add_argument("--out", default="results/audit")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.8)
    a = p.parse_args(argv)
    pt = power_table(pd.read_csv(a.gaps), a.alpha, a.power)
    out = Path(a.out)
    pt.to_csv(out / "power.csv", index=False)
    write_report(pt, out / "power.md")
    print((out / "power.md").read_text())


if __name__ == "__main__":
    main()
