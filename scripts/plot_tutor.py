"""Charts for the trained tutor (E5-E6), from saved files only (no GPU).

  python scripts/plot_tutor.py

Inputs: results/tutor/tutor_a/rewards.jsonl (every completion scored during GRPO; from notebook 03's
Output) and results/eval/{replies.jsonl, scores/*.jsonl} (notebook 05). If rewards.jsonl is missing, the
committed binned copy results/eval/training_curve.csv is used.

Outputs (results/eval/):
  training_curve.csv / .png   reward, grader scaffolding score and answer-leak rate during training
  base_vs_tutor.png           test-set metrics, base model vs trained tutor
  heldout_rm_by_leak.png      why the general-purpose RM scores the tutor lower: it rewards giving the answer
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("results/eval")
REWARDS = Path("results/tutor/tutor_a/rewards.jsonl")
BIN = 10  # training steps per point
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e5e5e5"
TUTOR, BASE = "#0b2a6f", "#9aa3b2"  # poster navy; grey for the untrained base model

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def training_curve() -> pd.DataFrame:
    if REWARDS.exists():
        r = pd.read_json(REWARDS, lines=True)
        c = (r.assign(bin=r["step"] // BIN * BIN + BIN / 2).groupby("bin")
             .agg(reward=("reward", "mean"), scaffolding=("scaffolding", "mean"),
                  answer_leaked=("answer_leaked", "mean"), n=("reward", "size")).reset_index())
        c.to_csv(OUT / "training_curve.csv", index=False)
        return c
    return pd.read_csv(OUT / "training_curve.csv")


def plot_training(c: pd.DataFrame) -> None:
    panels = [("reward", "Training reward (max 1.5)", None),
              ("scaffolding", "Grader: hint vs answer (-1 to +1)", None),
              ("answer_leaked", "Replies that leak the answer", "pct")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), dpi=200)
    for ax, (col, title, fmt) in zip(axes, panels):
        y = c[col] * (100 if fmt else 1)
        ax.plot(c["bin"], y, color=TUTOR, lw=2)
        ax.scatter(c["bin"].iloc[[0, -1]], y.iloc[[0, -1]], color=TUTOR, s=22, zorder=3)
        f = (lambda v: f"{v:.0f}%") if fmt else (lambda v: f"{v:+.2f}")
        title = f"{title}\nfirst 10 steps {f(y.iloc[0])}, last 10 steps {f(y.iloc[-1])}"
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("GRPO step")
        ax.grid(axis="y", color=GRID, lw=0.6)
        if fmt:
            ax.set_ylim(0, max(60, y.max() * 1.15))
    fig.tight_layout()
    fig.savefig(OUT / "training_curve.png")
    plt.close(fig)


def load_eval() -> pd.DataFrame:
    d = pd.read_json(OUT / "replies.jsonl", lines=True)
    j = pd.read_json(OUT / "scores/judge.jsonl", lines=True)[["id", "scaffolding", "helpfulness"]]
    d = d.merge(j.rename(columns={"id": "eval_id"}), on="eval_id")
    s = OUT / "scores/skywork_v2.jsonl"
    if s.exists():
        sk = pd.read_json(s, lines=True)[["id", "reward"]].rename(columns={"id": "eval_id", "reward": "skywork"})
        d = d.merge(sk, on="eval_id", how="left")
    return d


def bars(ax, labels, base_vals, tutor_vals, title, pct=False):
    x = range(len(labels))
    w = 0.38
    for off, vals, color, name in ((-w / 2 - 0.01, base_vals, BASE, "Base model"),
                                   (w / 2 + 0.01, tutor_vals, TUTOR, "Trained tutor")):
        b = ax.bar([i + off for i in x], vals, w, color=color, label=name)
        ax.bar_label(b, labels=[f"{v:.0f}%" if pct else f"{v:.1f}" for v in vals], padding=2, fontsize=8, color=INK)
    ax.set_xticks(list(x), labels)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_base_vs_tutor(d: pd.DataFrame) -> None:
    m = d.groupby("model").agg(scaffolds=("scaffolding", lambda s: (s > 0).mean() * 100),
                               leaked=("answer_leaked", lambda s: s.mean() * 100),
                               helpfulness=("helpfulness", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), dpi=200, gridspec_kw={"width_ratios": [2, 1]})
    bars(axes[0], ["Gives a hint\n(grader score > 0)", "Leaks the\nfinal answer"],
         [m.loc["base", "scaffolds"], m.loc["base", "leaked"]],
         [m.loc["tutor_a", "scaffolds"], m.loc["tutor_a", "leaked"]], "Share of test replies", pct=True)
    axes[0].set_ylim(0, 100)
    bars(axes[1], ["Helpfulness\n(1-5)"], [m.loc["base", "helpfulness"]], [m.loc["tutor_a", "helpfulness"]],
         "Grader helpfulness")
    axes[1].set_ylim(0, 5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "base_vs_tutor.png")
    plt.close(fig)


def plot_heldout(d: pd.DataFrame) -> None:
    if "skywork" not in d:
        return
    t = d.groupby(["model", "answer_leaked"])["skywork"].mean().unstack()
    fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=200)
    bars(ax, ["Hint only", "Leaks answer"], [t.loc["base", False], t.loc["base", True]],
         [t.loc["tutor_a", False], t.loc["tutor_a", True]], "Skywork RM score (held out)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "heldout_rm_by_leak.png")
    plt.close(fig)


def main() -> None:
    plot_training(training_curve())
    if (OUT / "replies.jsonl").exists():
        d = load_eval()
        plot_base_vs_tutor(d)
        plot_heldout(d)
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
