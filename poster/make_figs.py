"""Poster figure 3: score gap by student group for each grader (SVG, text set in the poster's font).

  python poster/make_figs.py     (needs the audit scores in results/audit/scores/ and audit_pairs.jsonl)

Rows: rewording control, three dialects, and "adds a name" (all name groups pooled vs no name).
Orange = penalty that survives the rewording control (dialects) or is clearly below 0 (names).
The pooled-name row is recomputed here; everything else matches results/audit/gaps.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from tutorbias.audit.analyze import drop_garbled, gap_table, load_scores  # noqa: E402
from tutorbias.io import load_config, read_jsonl  # noqa: E402

HERE = Path(__file__).resolve().parent
NAVY, ORANGE, MUTED = "#0b2a6f", "#c8501e", "#555555"
SCORERS = [("judge_scaffolding", "My grader"), ("skywork_v2", "Skywork RM"), ("tulu3_rm", "Tulu-3 RM")]
ROWS = [("paraphrase", "Rewording only"), ("aae", "African Am. English"), ("indian", "Indian English"),
        ("l2", "Learner English"), ("name_any", "Adds a name")]

plt.rcParams.update({"font.family": "Noto Sans", "svg.fonttype": "none", "font.size": 19,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})


def gaps() -> pd.DataFrame:
    cfg = load_config("configs/audit.yaml")
    pairs = drop_garbled(pd.DataFrame(read_jsonl(cfg["paths"]["pairs"])))
    pairs.loc[pairs["group"].str.startswith("name_"), "group"] = "name_any"
    df = load_scores(cfg, pairs)
    acfg = dict(cfg["analysis"], name_baseline_group=None)
    g = gap_table(df, [s for s, _ in SCORERS], acfg, cfg["seed"])
    return g[(g["subset"] == "all") & (g["vs"] == "mae")]


def main() -> None:
    g = gaps()
    g.to_csv(HERE / "fig_gaps.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True)
    for ax, (s, title) in zip(axes, SCORERS):
        d = g[g["scorer"] == s].set_index("group")
        for y, (grp, _) in enumerate(ROWS):
            if grp not in d.index:
                continue
            r = d.loc[grp]
            if grp == "name_any":
                bad = r["ci_hi"] < 0
            else:
                bad = pd.notna(r.get("net_ci_hi")) and r["net_ci_hi"] < 0
            c = ORANGE if bad else NAVY
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=c, lw=2.5, solid_capstyle="round")
            ax.scatter([r["gap_std"]], [y], color=c, s=55, zorder=3, edgecolor="white", linewidth=1.5)
        ax.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)))
        ax.set_title(title, fontsize=20, color=NAVY, fontweight="bold")
        ax.set_xlim(-0.15, 0.06)
        ax.set_xticks([-0.1, 0], ["-0.1", "0"])
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=MUTED)
        ax.grid(axis="x", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(ROWS)), [lab for _, lab in ROWS])
    axes[0].invert_yaxis()
    axes[1].set_xlabel("score change vs. Mainstream American English (SD)", color=MUTED)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(HERE / "fig_gaps.svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
    # use the poster's font stack for every label
    svg = (HERE / "fig_gaps.svg").read_text().replace("font-family:'Noto Sans'", "font-family:'Noto Sans',sans-serif")
    (HERE / "fig_gaps.svg").write_text(svg)
    print(g[["scorer", "group", "gap_std", "ci_lo", "ci_hi", "net_ci_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
