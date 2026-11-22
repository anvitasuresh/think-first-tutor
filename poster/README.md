# Expo poster (36 x 24 in, landscape)

- `template.html`: the poster text and layout (edit this).
- `build.py`: adds QR codes (live demo + repo) and renders `poster.html`, `poster.pdf` (36 x 24 in, print this) and `preview.png`.

```bash
pip install segno playwright && playwright install chromium
python poster/build.py          # prints the page height; it must say 2304 (nothing spills off the page)
```

**Status:** complete, no placeholders. Every number comes from `results/` (see the README results table).
The answer-vs-hint example in section 1 is written for the poster and labeled as such.

**Print file:** render `poster.pdf` at 200 dpi to a 7200 x 4800 JPG (`pdftoppm -r 200 -jpeg -jpegopt quality=95 poster/poster.pdf poster_36x24_print`) for print shops that want an image.
Check that both QR codes scan after every rebuild.

- `make_figs.py`: draws Fig. 2 (training) and Fig. 4 (score gaps) as SVG from the results files; run it before `build.py`.
