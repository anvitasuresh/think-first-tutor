# Expo poster (36 x 24 in, landscape)

- `template.html`: the poster text and layout (edit this).
- `build.py`: adds QR codes (live demo + repo) and renders `poster.html`, `poster.pdf` (36 x 24 in, print this) and `preview.png`.

```bash
pip install segno playwright && playwright install chromium
python poster/build.py          # prints the page height; it must say 2304 (nothing spills off the page)
```

**Before printing:** every orange dashed box is a result that is not in yet. Fill each one from the
named results file (audit: `results/audit/summary.md` and `forest_all.png`; explain: stage 3 outputs;
tutor: `results/eval/`), then rebuild. Never print a placeholder. The only real numbers on the poster
so far are the judge check (E1, `results/calibration/all/metrics.json`) and the grader's score on the
Fig. 1 reply (`results/judge/label_set_scores.jsonl`, L0160). The answer-vs-hint example in section 1
is written for the poster and labeled as such.
- `make_figs.py`: draws Fig. 2 (training) and Fig. 4 (score gaps) as SVG from the results files; run it before `build.py`.
