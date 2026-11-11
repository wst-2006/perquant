import pandas as pd

from scripts import derive_training_frame_v7
from src.quant_trading.ml_lab.research import MARKET_STATE_FEATURES


def test_derive_v7_replaces_normalized_macro_columns_with_raw_state(tmp_path, monkeypatch):
    source = tmp_path / "v5.csv"
    output = tmp_path / "v7.csv"
    rows = []
    for date, value in (("20250101", 10.0), ("20250102", 20.0)):
        for code in ("000001.SZ", "000002.SZ"):
            rows.append({"trade_date": date, "ts_code": code, "label_tradable": True, "feature_return_5d": value, **{column: 50.0 for column in MARKET_STATE_FEATURES}})
    pd.DataFrame(rows).to_csv(source, index=False, encoding="utf-8-sig")

    def raw_state(dates, factor_set):
        state = pd.DataFrame(
            {
                "trade_date": dates,
                **{column.replace("feature_", "feature_state_", 1): [index + 0.1 for index, _ in enumerate(dates)] for column in MARKET_STATE_FEATURES},
            }
        )
        return state, {"source": "test", "factor_set": factor_set}

    monkeypatch.setattr(derive_training_frame_v7, "_raw_state_by_date", raw_state)
    result = derive_training_frame_v7.derive_frame(source, output, chunksize=2)
    frame = pd.read_csv(output)

    assert result["raw_state_coverage"]["feature_state_market_return_1d"] == 1.0
    assert not set(MARKET_STATE_FEATURES) & set(frame.columns)
    assert frame.groupby("trade_date")["feature_state_market_return_1d"].first().tolist() == [0.1, 1.1]
