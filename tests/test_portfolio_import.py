from pathlib import Path

from src.quant_trading.services.portfolio import PortfolioService


class DummyClient:
    pass


def test_parse_buy_date_variants():
    service = PortfolioService(DummyClient())
    assert service._parse_buy_date("2026-06-18").isoformat() == "2026-06-18"
    assert service._parse_buy_date("20260618").isoformat() == "2026-06-18"


def test_import_csv_missing_file_raises(tmp_path):
    service = PortfolioService(DummyClient())
    missing = tmp_path / "missing.csv"
    try:
        service.import_positions_csv(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("FileNotFoundError expected")
