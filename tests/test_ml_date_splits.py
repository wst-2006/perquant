from src.quant_trading.ml_lab.date_splits import REPRESENTATIVE_A_SHARE_20240401_20260601_480, split_trade_dates


def test_representative_split_keeps_holdout_out_of_training():
    dates = ["20240927", "20240930"] + [f"202410{day:02d}" for day in range(8, 26)] + ["20241101"]
    result = split_trade_dates(dates, REPRESENTATIVE_A_SHARE_20240401_20260601_480)
    assert "20241008" in result["holdout_dates"]
    assert "20241025" in result["holdout_dates"]
    assert "20241008" not in result["train_dates"]
    assert "20240927" not in result["train_dates"]
    assert "20240930" not in result["train_dates"]
    assert result["label_blocked_train_dates"] == ["20240927", "20240930"]
    assert "20241101" in result["train_dates"]
    assert result["leakage_policy"].startswith("Holdout signal dates")
