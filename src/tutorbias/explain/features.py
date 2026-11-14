"""E3, part 1: which dialect features move the scores? (CPU, runs on the stage 2 audit scores)

  python -m tutorbias.explain.features --config configs/explain.yaml
  # -> results/explain/feature_effects.csv, feature_effects.png

Every dialect variant records which Multi-VALUE features fired in it. For each (reply, variant) pair:

  delta = score(reply, variant) - score(reply, MAE message)      in SD units of that scorer's MAE scores

and we regress delta on feature indicators (one 0/1 column per feature, plus an intercept for "the
message changed at all"). Because features are applied at random (each allowed feature with p = 0.5,
independently per message), which features a message got is close to randomized, so the coefficients
estimate each feature's effect with little confounding. Features shared by groups (e.g. 3sg -s
absence in AAE and L2 English) are pooled, which is the point: groups overlap, features don't.

Uncertainty: bootstrap over questions (refit on resampled questions). Control: a permutation null,
shuffling the feature sets across pairs within each group and refitting; a feature counts only if its
effect is larger than the permutation's 95th percentile of |beta| (a "random feature" baseline that
also handles multiple comparisons).

Limitation: this is attribution from randomized co-occurrence, not single-feature re-generation. The
stretch version re-transforms each message with one feature removed (needs the Multi-VALUE env).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..audit.analyze import load_scores
from ..io import load_config, read_jsonl


def pair_deltas(df: pd.DataFrame, scorer: str, baseline: str = "mae") -> pd.DataFrame:
    """One row per (reply, non-MAE dialect variant): delta vs the same reply with the MAE message."""
    d = df[df["axis"] == "dialect"]
    base = d[d["group"] == baseline].set_index("reply_id")[scorer]
    sd = float(np.nanstd(base)) or 1.0
    v = d[d["group"] != baseline].copy()
    v["delta"] = (v[scorer].to_numpy(float) - base.reindex(v["reply_id"]).to_numpy(float)) / sd
    return v.dropna(subset=["delta"])


def design(rows: pd.DataFrame, min_count: int) -> tuple[np.ndarray, list[str]]:
    counts: dict[str, int] = {}
    for f in rows["features"]:
        for k in (f or {}):
            counts[k] = counts.get(k, 0) + 1
    feats = sorted(k for k, c in counts.items() if c >= min_count)
    x = np.zeros((len(rows), len(feats) + 1))
    x[:, 0] = rows["changed"].astype(float).to_numpy()
    idx = {f: i + 1 for i, f in enumerate(feats)}
    for r, f in enumerate(rows["features"]):
        for k in (f or {}):
            if k in idx:
                x[r, idx[k]] = 1.0
    return x, ["(any change)"] + feats


def fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Least squares with a tiny ridge (rare features can be collinear with each other)."""
    xtx = x.T @ x + ridge * np.eye(x.shape[1])
    return np.linalg.solve(xtx, x.T @ y)


def feature_effects(v: pd.DataFrame, min_count: int, n_boot: int, n_perm: int, seed: int,
                    alpha: float = 0.05) -> pd.DataFrame:
    x, names = design(v, min_count)
    y = v["delta"].to_numpy(float)
    beta = fit(x, y)
    rng = np.random.default_rng(seed)
    items = v["item_id"].to_numpy()
    codes, uniq = pd.factorize(items)
    rows_of = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    boots = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        pick = np.concatenate([rows_of[k] for k in rng.integers(0, len(uniq), len(uniq))])
        boots[b] = fit(x[pick], y[pick])
    # Permutation null: shuffle feature rows within each group, keep the "changed" column.
    groups = v["group"].to_numpy()
    null_max = np.empty(n_perm)
    for b in range(n_perm):
        xp = x.copy()
        for g in np.unique(groups):
            m = np.flatnonzero(groups == g)
            xp[m, 1:] = x[rng.permutation(m), 1:]
        null_max[b] = np.max(np.abs(fit(xp, y)[1:])) if len(names) > 1 else 0.0
    thresh = float(np.quantile(null_max, 1 - alpha)) if n_perm else float("nan")
    n_with = x.sum(0)
    out = pd.DataFrame({
        "feature": names, "beta": beta, "ci_lo": np.quantile(boots, alpha / 2, axis=0),
        "ci_hi": np.quantile(boots, 1 - alpha / 2, axis=0), "n_pairs_with": n_with.astype(int),
        "perm_threshold": thresh,
    })
    out["beyond_null"] = (out["beta"].abs() > thresh) & (out["feature"] != "(any change)")
    out["groups"] = [", ".join(sorted({g for g, f in zip(groups, v["features"]) if f and n in f})) or "all"
                     for n in names]
    return out.sort_values("beta").reset_index(drop=True)


def plot(eff: pd.DataFrame, path: Path, top: int = 20) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorers = list(dict.fromkeys(eff["scorer"]))
    fig, axes = plt.subplots(1, len(scorers), figsize=(5 * len(scorers), 0.3 * top + 1.5), squeeze=False)
    for ax, s in zip(axes[0], scorers):
        e = eff[(eff["scorer"] == s) & (eff["feature"] != "(any change)")]
        e = e.reindex(e["beta"].abs().sort_values(ascending=False).index).head(top).sort_values("beta")
        colors = ["C3" if b else "grey" for b in e["beyond_null"]]
        ax.barh(e["feature"], e["beta"], xerr=[e["beta"] - e["ci_lo"], e["ci_hi"] - e["beta"]], color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(s)
        ax.set_xlabel("effect on score (SD units), 95% CI")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/explain.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    acfg = load_config(cfg["audit_config"])
    fcfg = cfg["features"]
    pairs = pd.DataFrame(read_jsonl(acfg["paths"]["pairs"]))
    df = load_scores(acfg, pairs)
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for s in [s for s in fcfg["scorers"] if s in df]:
        v = pair_deltas(df, s)
        e = feature_effects(v, fcfg["min_count"], fcfg["n_boot"], fcfg["n_perm"], cfg["seed"])
        e.insert(0, "scorer", s)
        tables.append(e)
        print(f"[features] {s}: {int(e['beyond_null'].sum())} features beyond the permutation null")
    if not tables:
        print("[features] no audit scores found; run the audit stage first")
        return
    eff = pd.concat(tables, ignore_index=True)
    eff.to_csv(out / "feature_effects.csv", index=False)
    plot(eff, out / "feature_effects.png")
    print(f"[features] wrote {out / 'feature_effects.csv'}")


if __name__ == "__main__":
    main()
