import pandas as pd

from scripts.derive_training_frame_v3 import derive_frame
from src.quant_trading.ml_lab.service import MlLabService


def test_derive_v3_frame_drops_contract_failed_columns(tmp_path):
    source = tmp_path / "v2.csv"
    output = tmp_path / "v3.csv"
    rows = []
    feature_names = MlLabService.feature_names("factor_lab_v3")
    for index in range(4):
        row = {
            "trade_date": f"2025010{index + 1}",
            "ts_code": f"00000{index}.SZ",
            "label_tradable": True,
            "label_net_return": 0.01,
            "label_stable_positive": 1,
            "label_top5": 0,
            "label_top10": 1,
            "feature_assets_yoy": None,
        }
        for name in feature_names:
            row[f"feature_{name}"] = float(index + 1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(source, index=False, encoding="utf-8-sig")

    result = derive_frame(source, output, feature_set="factor_lab_v3", chunksize=2)
    frame = pd.read_csv(output)

    assert result["rows"] == 4
    assert "feature_assets_yoy" not in frame.columns
    assert "feature_return_120d" in frame.columns
    assert result["feature_missingness_gate"]["status"] == "pass"


def test_derive_v4_frame_builds_real_derived_meta_features(tmp_path):
    source = tmp_path / "v3.csv"
    output = tmp_path / "v4.csv"
    rows = []
    feature_names = MlLabService.feature_names("factor_lab_v4")
    base_names = [name for name in feature_names if not name.endswith("risk_adjusted") and name not in {"short_vs_medium_return", "medium_vs_long_return", "volatility_change_5d_20d", "amount_trend_5d_20d", "range_pressure_20d", "limit_pressure_5d", "moneyflow_persistence_5d_10d", "flow_momentum_alignment_5d_20d"}]
    for index in range(4):
        row = {
            "trade_date": f"2025010{index + 1}",
            "ts_code": f"00000{index}.SZ",
            "label_tradable": True,
            "label_net_return": 0.01,
            "label_stable_positive": 1,
            "label_top5": 0,
            "label_top10": 1,
        }
        for name in base_names:
            row[f"feature_{name}"] = float(index + 2)
        row["feature_return_5d"] = 0.05
        row["feature_return_20d"] = 0.02
        row["feature_return_60d"] = 0.01
        row["feature_volatility_5d"] = 0.03
        row["feature_volatility_20d"] = 0.02
        row["feature_amount_avg_5d"] = 120.0
        row["feature_amount_avg_20d"] = 100.0
        row["feature_distance_to_20d_high"] = -0.1
        row["feature_distance_to_20d_low"] = 0.2
        row["feature_limit_up_count_5d"] = 2.0
        row["feature_limit_down_count_5d"] = 1.0
        row["feature_moneyflow_positive_days_5d"] = 3.0
        row["feature_moneyflow_positive_days_10d"] = 5.0
        row["feature_relative_industry_strength_20d"] = 0.04
        row["feature_main_net_to_amount_sum_5d"] = 0.01
        row["feature_main_net_to_circ_mv_sum_5d"] = 0.001
        row["feature_value_composite"] = 0.7
        rows.append(row)
    pd.DataFrame(rows).to_csv(source, index=False, encoding="utf-8-sig")

    result = derive_frame(source, output, feature_set="factor_lab_v4", chunksize=2)
    frame = pd.read_csv(output)

    assert result["rows"] == 4
    assert "feature_roe" not in frame.columns
    assert frame["feature_short_vs_medium_return"].iloc[0] == 0.03
    assert frame["feature_industry_strength_risk_adjusted"].iloc[0] == 2.0
    assert result["feature_missingness_gate"]["status"] == "pass"


def test_derive_v5_frame_keeps_cross_sectional_ranks_intact_across_chunks(tmp_path):
    source = tmp_path / "v4.csv"
    output = tmp_path / "v5.csv"
    v4_names = MlLabService.feature_names("factor_lab_v4")
    rows = []
    for trade_date, base in (("20250101", 1.0), ("20250102", 10.0)):
        for stock, multiplier in (("000001.SZ", 1.0), ("000002.SZ", 2.0)):
            row = {
                "trade_date": trade_date,
                "ts_code": stock,
                "label_tradable": True,
                "label_net_return": 0.01,
                "label_stable_positive": 1,
                "label_top5": 0,
                "label_top10": 1,
            }
            for name in v4_names:
                row[f"feature_{name}"] = base * multiplier
            row["feature_moneyflow_positive_days_5d"] = 3.0
            row["feature_moneyflow_positive_days_10d"] = 5.0
            row["feature_downside_volatility_20d"] = 0.02
            rows.append(row)
    pd.DataFrame(rows).to_csv(source, index=False, encoding="utf-8-sig")

    result = derive_frame(source, output, feature_set="factor_lab_v5", chunksize=3)
    frame = pd.read_csv(output)

    assert result["feature_missingness_gate"]["status"] == "pass"
    assert result["feature_count"] > len(v4_names)
    assert frame.groupby("trade_date")["feature_cs_rank_return_5d"].apply(list).tolist() == [[0.5, 1.0], [0.5, 1.0]]
    assert frame.groupby("trade_date")["feature_market_breadth_return_5d"].nunique().eq(1).all()
