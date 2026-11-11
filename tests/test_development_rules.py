from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_production_code_does_not_filter_full_market_stock_pool():
    offenders = []
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "get_all_a_shares(with_filter=True)" in text:
            offenders.append(path.as_posix())
    assert offenders == []
