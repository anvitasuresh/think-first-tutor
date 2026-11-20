"""Build the Expo poster: QR codes -> poster.html -> poster.pdf (24 x 24 in) + preview.png.

Usage:  pip install segno playwright && python poster/build.py
Edit poster/template.html for text; boxes marked class="pending" are results still to fill in.
"""

import os
from pathlib import Path

import segno
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
QR = {
    "QR_DEMO": "https://anvitasuresh.github.io/xai_proj2/",
    "QR_REPO": "https://github.com/anvitasuresh/xai_proj2",
}


def qr_svg(url: str) -> str:
    """Inline SVG QR code, navy on white, scales to its container."""
    svg = segno.make(url, error="m").svg_inline(scale=10, border=2, dark="#0b2a6f", light="#ffffff")
    return svg.replace("<svg ", '<svg class="qr" ', 1)


def main() -> None:
    html = (HERE / "template.html").read_text()
    for key, url in QR.items():
        html = html.replace("{{" + key + "}}", qr_svg(url))
    # Fig. 3 is inlined (an <img> SVG cannot use the page's web fonts). Made by poster/make_figs.py.
    fig = HERE / "fig_gaps.svg"
    if fig.exists():
        svg = fig.read_text()
        html = html.replace("{{FIG_GAPS}}", svg[svg.index("<svg"):].replace("<svg ", '<svg style="width:100%;height:auto" ', 1))
    out = HERE / "poster.html"
    out.write_text(html)

    with sync_playwright() as p:
        # CHROMIUM=/path/to/chrome uses a specific browser binary instead of the Playwright download
        exe = os.environ.get("CHROMIUM")
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe else p.chromium.launch()
        # 24 in x 24 in at 96 CSS px per inch = 2304 px square
        page = browser.new_page(viewport={"width": 2304, "height": 2304}, device_scale_factor=1)
        page.goto(out.as_uri())
        page.wait_for_timeout(1500)  # let web fonts load
        page.pdf(path=str(HERE / "poster.pdf"), width="24in", height="24in", print_background=True)
        page.screenshot(path=str(HERE / "preview.png"), full_page=False)
        overflow = page.evaluate("document.documentElement.scrollHeight")
        browser.close()
    print("wrote poster.html, poster.pdf, preview.png; page height", overflow, "(should be 2304)")


if __name__ == "__main__":
    main()
