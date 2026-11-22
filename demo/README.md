# Interactive demo

A single web page for visitors (the Expo crowd, teachers, anyone new to the project). It is meant to be
played with; the detailed numbers live in the main README. Five parts:

1. **Can you get the answer out of it?** Pick what to say and the trained tutor replies, with a toggle to
   see what the untrained base model said to the same message and a running count of how often you got
   the answer. Replies are real ones recorded during evaluation (Mainstream US messages, replies cut off by
   the 256-token limit skipped, 24 per message type sampled with a fixed seed). It is not a live model.
2. **How the tutor learned.** A click-through of one GRPO step (four illustrative tries, clearly labeled
   as an illustration), then the real training curve and before/after numbers.
3. **Hint or answer?** Guess whether a real tutor reply guides the student or gives the work away, then
   compare with Anvita's label and the AI grader's score and explanation (the 40-item human spot check, E1).
4. **Same help, different student.** One tutor reply next to every version of the student's message
   (AAE, Indian English, learner English, a reworded control, four name cues), with the grader's score.
5. **What I found.** Four plain-language takeaways.

It needs no server, GPU or API key. Everything is baked into `site/index.html`.

## Rebuild after a Kaggle stage finishes

```bash
python demo/build.py        # standard library only; reads data/, results/ and demo/variants_sample.jsonl
git add demo && git commit -m "Demo: refresh with new results" && git push
```

`build.py` fills each part from whatever exists:

| Part | Reads |
|---|---|
| 1 | `results/eval/replies.jsonl`, `results/eval/scores/judge.jsonl` |
| 2 | `results/eval/training_curve.csv`, plus the two files above for the before/after numbers |
| 3 | `data/labels/*`, `results/judge/label_set_scores.jsonl`, `results/calibration/all/metrics.json` |
| 4 | `data/processed/variants.jsonl` if present (copied into `demo/variants_sample.jsonl`), else the committed copy; per-pair scores in `results/audit/scores/` with `data/processed/audit_pairs.jsonl` |

The eval and audit score files are gitignored, so rebuild from a checkout that has them (the committed
`site/index.html` already has the data baked in). Apart from the GRPO illustration, which says so on the
page, the page never shows a number that a run did not produce.

## Known variant glitches

Multi-VALUE's `aint_before_main` sometimes turns a positive "do/did" into a bare "ai" ("what I ai wrong"),
and `past_for_past_participle` turns adjectival "done" into "did" ("I am did with it"). The audit drops
these variants (`tutorbias.textutil.garbled_reason`), and the demo uses the same check.

## Publishing

`.github/workflows/pages.yml` deploys `demo/site/` to GitHub Pages on every push to `main` that touches
`demo/`. One-time setup: repo **Settings > Pages > Source: GitHub Actions**. The page is then at
https://anvitasuresh.github.io/think-first-tutor/. To view it locally, open `demo/site/index.html` in a browser.
