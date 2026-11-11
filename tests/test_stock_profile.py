from src.quant_trading.services.stock_profile import StockProfileService


def test_stock_profile_radar_uses_available_real_factor_scores_only():
    radar = StockProfileService.build_radar(
        {
            "return_5d": 80.0,
            "return_20d": 60.0,
            "main_net_to_amount": 70.0,
            "relative_industry_strength_20d": 90.0,
        }
    )

    dimensions = {item["key"]: item for item in radar}
    assert len(radar) == 6
    assert dimensions["momentum"]["value"] == 70.0
    assert dimensions["capital_flow"]["value"] == 70.0
    assert dimensions["quality"]["value"] is None


def test_stock_profile_interpretation_is_descriptive_not_a_promotion():
    radar = [
        {"key": "momentum", "value": 80.0},
        {"key": "capital_flow", "value": 70.0},
        {"key": "industry", "value": 75.0},
        {"key": "risk", "value": 65.0},
        {"key": "quality", "value": 70.0},
    ]

    result = StockProfileService.interpret(radar, stable_score=75.0, aggressive_score=72.0)

    assert result["strong_state"] == "强势"
    assert result["research_fit_state"] == "可继续观察"
    assert "not a v11 promotion gate" in result["interpretation_policy"]
