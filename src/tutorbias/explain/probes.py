"""E3, part 2: linear probes on the judge's activations.

  python -m tutorbias.explain.probes extract --config configs/explain.yaml   # GPU: hidden states
  python -m tutorbias.explain.probes fit     --config configs/explain.yaml   # CPU: probes + plots

Extract. For each pair in the E4 subset (same replies, MAE and each dialect version of the message),
run the judge's exact scoring prompt (rubric + problem + student + reply + "SCAFFOLDING: ") and keep
the hidden state of the last token, the position where the judge reads off its score, at a set of
layers. The 7B judge runs in fp16 split over both T4s (device_map="auto").

Fit, per layer:
  1. Dialect probe: logistic regression, variant vs MAE, cross-validated with questions grouped (no
     question in both train and test). Control: the same probe on shuffled labels. High accuracy means
     the judge represents the student's dialect at the scoring position, which is necessary (not
     sufficient) for the score to depend on it.
  2. Alignment: the "dialect direction" (mean variant activation - mean MAE activation) vs the
     "score direction" (ridge regression of the judge's own scaffolding score on activations, fit on
     MAE rows). Cosine similarity, and the Spearman correlation between each variant's activation shift
     projected on the score direction and its actual score gap. A clear negative alignment is a
     mechanistic account of the E2 gap: dialect moves the representation toward "gives the answer".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io import load_config, md_table, read_jsonl


def extract(cfg: dict[str, Any]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..judge.judge import build_conv, load_rubric

    pc, paths = cfg["probes"], cfg["paths"]
    jcfg = load_config(cfg["judge_config"])["judges"]["primary"]
    model_name = jcfg["model"]["model"]
    rubric = load_rubric(jcfg.get("rubric", "v1"))
    rows = read_jsonl(paths["subset"])
    out = Path(paths["activations"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[probes] {out} exists; delete it to re-extract")
        return
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, pc.get("dtype", "float16")), device_map="auto").eval()
    n_layers = model.config.num_hidden_layers
    layers = pc.get("layers") or list(range(0, n_layers + 1, max(1, n_layers // 7)))
    feats = {layer: [] for layer in layers}
    bs = pc.get("batch_size", 4)
    for i in range(0, len(rows), bs):
        batch = rows[i : i + bs]
        texts = [tok.apply_chat_template(build_conv(rubric, r), tokenize=False, add_generation_prompt=True)
                 + "SCAFFOLDING: " for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        for layer in layers:
            feats[layer].append(hs[layer][:, -1, :].float().cpu().numpy().astype(np.float16))
        if (i // bs) % 25 == 0:
            print(f"[probes] {i + len(batch)}/{len(rows)}")
    np.savez_compressed(out, layers=np.array(layers), pair_id=np.array([r["pair_id"] for r in rows]),
                        **{f"L{layer}": np.concatenate(v) for layer, v in feats.items()})
    print(f"[probes] saved {out} ({len(rows)} rows, layers {layers})")


def grouped_cv_accuracy(x: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    accs = []
    for tr, te in GroupKFold(n_splits=folds).split(x, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=seed))
        clf.fit(x[tr], y[tr])
        accs.append(float((clf.predict(x[te]) == y[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def alignment(x: np.ndarray, meta: pd.DataFrame, group: str) -> dict[str, float]:
    """Cosine(dialect direction, score direction) and Spearman(projected shift, actual score gap)."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge

    mae = meta["group"].to_numpy() == "mae"
    var = meta["group"].to_numpy() == group
    score_dir = Ridge(alpha=10.0).fit(x[mae], meta.loc[mae, "scaffolding"].to_numpy(float)).coef_
    score_dir = score_dir / (np.linalg.norm(score_dir) or 1.0)
    mae_idx = pd.Series(np.flatnonzero(mae), index=meta.loc[mae, "reply_id"].to_numpy())
    vi = np.flatnonzero(var)
    keep = [i for i in vi if meta["reply_id"].iat[i] in mae_idx.index]
    if len(keep) < 5:
        return {"cosine": float("nan"), "spearman_shift_vs_gap": float("nan"), "n": len(keep)}
    mi = mae_idx.loc[meta["reply_id"].iloc[keep].to_numpy()].to_numpy()
    shift = x[keep] - x[mi]
    dialect_dir = shift.mean(0)
    cos = float(dialect_dir @ score_dir / (np.linalg.norm(dialect_dir) or 1.0))
    gap = meta["scaffolding"].to_numpy(float)[keep] - meta["scaffolding"].to_numpy(float)[mi]
    rho = spearmanr(shift @ score_dir, gap).statistic if np.std(gap) > 0 else float("nan")
    return {"cosine": cos, "spearman_shift_vs_gap": float(rho), "n": len(keep)}


def fit(cfg: dict[str, Any]) -> None:
    pc, paths = cfg["probes"], cfg["paths"]
    z = np.load(paths["activations"])
    meta = pd.DataFrame({"pair_id": z["pair_id"]})
    sub = pd.DataFrame(read_jsonl(paths["subset"]))[["pair_id", "reply_id", "item_id", "group"]]
    meta = meta.merge(sub, on="pair_id", how="left")
    sc = pd.DataFrame(read_jsonl(paths["rationales"]))[["id", "scaffolding"]].rename(columns={"id": "pair_id"})
    meta = meta.merge(sc.drop_duplicates("pair_id"), on="pair_id", how="left")
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for layer in z["layers"]:
        x = z[f"L{layer}"].astype(np.float32)
        for g in [g for g in meta["group"].unique() if g not in ("mae", "paraphrase")]:
            m = meta["group"].isin(["mae", g]).to_numpy()
            y = (meta.loc[m, "group"] == g).to_numpy(int)
            grp = meta.loc[m, "item_id"].to_numpy()
            acc = grouped_cv_accuracy(x[m], y, grp, pc["folds"], cfg["seed"])
            acc_null = grouped_cv_accuracy(x[m], rng.permutation(y), grp, pc["folds"], cfg["seed"])
            al = alignment(x, meta, g) if meta["scaffolding"].notna().all() else {}
            rows.append({"layer": int(layer), "group": g, "probe_acc": acc, "shuffled_acc": acc_null, **al})
        print(f"[probes] layer {layer} done")
    res = pd.DataFrame(rows)
    out = Path(paths["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "probes.csv", index=False)
    plot(res, out / "probes.png")
    (out / "probes.md").write_text("# E3: linear probes on the judge\n\n" + md_table(res) + "\n\n![probes](probes.png)\n")
    print(json.dumps(res.groupby("group")["probe_acc"].max().to_dict(), indent=2))


def plot(res: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for k, (g, r) in enumerate(res.groupby("group")):
        axes[0].plot(r["layer"], r["probe_acc"], marker="o", color=f"C{k}", label=g)
        axes[0].plot(r["layer"], r["shuffled_acc"], ls="--", color=f"C{k}", alpha=0.5)
        if "cosine" in r:
            axes[1].plot(r["layer"], r["cosine"], marker="o", color=f"C{k}", label=g)
    axes[0].set(xlabel="layer", ylabel="accuracy (dashed: shuffled labels)", title="Is dialect decodable?")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set(xlabel="layer", ylabel="cosine", title="Dialect direction vs score direction")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "fit"])
    ap.add_argument("--config", default="configs/explain.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    extract(cfg) if a.cmd == "extract" else fit(cfg)


if __name__ == "__main__":
    main()
