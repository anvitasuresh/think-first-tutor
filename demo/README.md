# Interactive demo

A single web page that shows the project to someone who has never heard of it, in three parts:

1. **Hint or answer?** Guess whether a real tutor reply guides the student or gives the work away, then
   compare with Anvita's label and the AI judge's score and explanation (the 40-item human spot check, E1).
2. **Same help, different student.** One tutor reply next to every version of the student's message:
   AAE, Indian English, learner English, a reworded control, and four name cues. Changed words are
   highlighted and the eWAVE grammar features are listed. Reward-model gaps appear once the audit (E2) has run.
3. **What the tutor learned.** Base model vs the GRPO tutor on the same held-out question (E5-E7),
   once stage 5 has run.

It needs no server, GPU or API key. Everything is baked into `site/index.html`.

## Rebuild after a Kaggle stage finishes

```bash
python demo/build.py        # standard library only; reads data/, results/ and demo/variants_sample.jsonl
git add demo && git commit -m "Demo: refresh with new results" && git push
```

`build.py` fills each part from whatever exists:

| Part | Reads | If missing |
|---|---|---|
| 1 | `data/labels/*`, `results/judge/label_set_scores.jsonl`, `results/calibration/all/metrics.json` | (always present) |
| 2, messages | `data/processed/variants.jsonl` if present (copied into `demo/variants_sample.jsonl`), else the committed copy | |
| 2, scores | `results/audit/gaps.csv` (and per-pair scores in `results/audit/scores/` with `data/processed/audit_pairs.jsonl`) | "Result pending" |
| 3 | `results/eval/replies.jsonl`, `results/eval/scores/judge.jsonl`, `results/eval/metrics_by_group.csv` | "Result pending" |

The page never shows a number that a run did not produce.

## Known variant glitches

Multi-VALUE's `aint_before_main` sometimes turns a positive "do/did" into a bare "ai" ("what I ai wrong"),
and `past_for_past_participle` turns adjectival "done" into "did" ("I am did with it"). About 20% and 7%
of the AAE variants have these. The demo skips them (`GLITCHES` in `build.py`); the audit should
exclude them too before its AAE results are reported.

## Publishing

`.github/workflows/pages.yml` deploys `demo/site/` to GitHub Pages on every push to `main` that touches
`demo/`. One-time setup: repo **Settings > Pages > Source: GitHub Actions**. The page is then at
https://anvitasuresh.github.io/xai_proj2/. To view it locally, open `demo/site/index.html` in a browser.
