import pandas as pd

from src.quant_trading.factor_lab.metrics import ic_ir, max_drawdown, rank_ic, verdict_from_metrics, winsorized_zscore


def test_rank_ic_uses_rank_correlation():
    value = rank_ic([1, 2, 3, 4], [0.01, 0.02, 0.03, 0.04])
    assert value == 1.0


def test_rank_ic_rejects_constant_series():
    assert rank_ic([1, 1, 1], [0.01, 0.02, 0.03]) is None


def test_ic_ir_requires_variation():
    assert ic_ir([0.01, 0.03, 0.05]) > 0
    assert ic_ir([0.01, 0.01, 0.01]) is None


def test_max_drawdown_compounds_returns():
    assert max_drawdown([0.1, -0.2, 0.05]) < 0


def test_winsorized_zscore_preserves_missing_values():
    scores = winsorized_zscore(pd.Series([1, 2, None, 100]))
    assert pd.isna(scores.iloc[2])
    assert round(float(scores.dropna().mean()), 10) == 0


def test_verdict_from_metrics():
    assert verdict_from_metrics(0.04, 0.3, 0.01, True) == "可用"
    assert verdict_from_metrics(0.01, 0.1, 0.01, False) == "观察"
    assert verdict_from_metrics(-0.01, 0.1, 0.01, True) == "弃用"
