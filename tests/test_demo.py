"""The demo page builds from the committed files alone and never shows numbers that don't exist yet."""

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_build():
    spec = importlib.util.spec_from_file_location("demo_build", ROOT / "demo" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_game_uses_the_40_human_labels():
    game = _load_build().build_game()
    assert len(game["rounds"]) == 40
    assert {r["human"] for r in game["rounds"]} <= {-1, 0, 1}
    assert all(-1 <= r["judge"] <= 1 for r in game["rounds"])


def test_glitch_filter_catches_broken_aint():
    b = _load_build()
    assert any(p.search("I don't see what I ai wrong.") for p in b.GLITCHES)
    assert any(p.search("I am did with it.") for p in b.GLITCHES)
    assert not any(p.search("I ain't know where to start.") for p in b.GLITCHES)


def test_page_builds_and_embeds_data(tmp_path, monkeypatch):
    b = _load_build()
    monkeypatch.setattr(b, "OUT", tmp_path / "index.html")
    b.main()
    html = (tmp_path / "index.html").read_text()
    assert "/*__DATA__*/" not in html
    data = json.loads(re.search(r"const D = (.*);\n", html).group(1).replace("<\\/", "</"))
    for ex in data["voices"]["examples"]:
        assert "mae" in ex["variants"]
        for v in ex["variants"].values():
            assert not any(p.search(v["text"]) for p in b.GLITCHES)
