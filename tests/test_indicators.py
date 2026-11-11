import pandas as pd

from src.quant_trading.analysis.indicators import technical_score


def test_technical_score_returns_valid_score_for_sufficient_data():
    rows = []
    for i in range(80):
        price = 10 + i * 0.1
        rows.append(
            {
                "trade_date": f"202501{i + 1:02d}",
                "open": price - 0.05,
                "high": price + 0.1,
                "low": price - 0.1,
                "close": price,
                "vol": 1000 + i * 5,
            }
        )
    result = technical_score(pd.DataFrame(rows))
    assert result["valid"] is True
    assert 0 <= result["score"] <= 100


def test_technical_score_rejects_insufficient_data():
    result = technical_score(pd.DataFrame([{"trade_date": "20250101", "close": 10, "high": 10, "low": 10}]))
    assert result["valid"] is False
    assert result["score"] == 0.0
