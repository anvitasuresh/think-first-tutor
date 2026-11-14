"""Kaggle helpers: where restored files land, which copy wins, and the GPU check."""

from pathlib import Path

from tutorbias.kaggle import check_gpus, destination, plan_restore, restore


def test_destination_rules():
    assert destination(Path("xai_proj2/data/processed/variants.jsonl")) == Path("data/processed/variants.jsonl")
    assert destination(Path("xai_proj2/results/audit/scores/judge.jsonl")) == Path("results/audit/scores/judge.jsonl")
    # laptop upload of data/processed with files at the top level
    assert destination(Path("items_audit.jsonl")) == Path("data/processed/items_audit.jsonl")
    assert destination(Path("label_set.jsonl")) == Path("data/labels/label_set.jsonl")
    assert destination(Path("xai_proj2/.git/config")) is None
    assert destination(Path("xai_proj2/src/tutorbias/io.py")) is None
    assert destination(Path("random.txt")) is None


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_restore_keeps_largest_copy(tmp_path):
    inp, repo = tmp_path / "input", tmp_path / "repo"
    _write(inp / "nb01-output/xai_proj2/results/audit/scores/judge.jsonl", "a\n")
    _write(inp / "nb02-partial/xai_proj2/results/audit/scores/judge.jsonl", "a\nb\nc\n")
    _write(inp / "tutorbias-data/items_audit.jsonl", "{}\n")
    _write(repo / "data/processed/items_audit.jsonl", "{}\n{}\n")  # repo copy is larger: keep it
    plan = plan_restore(inp, repo)
    dsts = {d.relative_to(repo): s for s, d in plan}
    assert Path("results/audit/scores/judge.jsonl") in dsts
    assert "nb02-partial" in str(dsts[Path("results/audit/scores/judge.jsonl")])
    assert Path("data/processed/items_audit.jsonl") not in dsts
    assert restore(inp, repo) == 1
    assert (repo / "results/audit/scores/judge.jsonl").read_text() == "a\nb\nc\n"
    assert restore(inp, repo) == 0  # idempotent


def test_gpu_check():
    assert check_gpus(["Tesla T4", "Tesla T4"])[0]
    ok, msg = check_gpus(["Tesla P100-PCIE-16GB"])
    assert not ok and "T4 x2" in msg
    assert not check_gpus([])[0]
