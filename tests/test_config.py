import pytest

from src.quant_trading.config import CapitalAllocation, StrategySettings, settings


def test_reference_capital_allocation_sums_to_one_but_is_optional():
    settings.strategy.allocation.validate()
    assert settings.strategy.allocation.short_term_core == 0.60
    assert settings.strategy.allocation.aggressive == 0.30
    assert settings.strategy.allocation.cash == 0.10
    assert settings.strategy.capital_mode == "manual"
    assert settings.strategy.allocation_used_for_selection is False


def test_factor_weights_sum_to_one():
    settings.strategy.validate()
    assert abs(sum(settings.strategy.factor_weights.values()) - 1.0) < 1e-9


def test_preliminary_filter_not_shrunk():
    assert settings.strategy.preliminary_filter_top_n >= 2000


def test_invalid_allocation_raises():
    with pytest.raises(Exception):
        CapitalAllocation(short_term_core=0.5, aggressive=0.3, cash=0.3).validate()


def test_invalid_top_n_raises():
    config = StrategySettings(preliminary_filter_top_n=1000)
    with pytest.raises(Exception):
        config.validate()


def test_public_config_uses_modern_zero_one_weights():
    public = settings.as_public_dict()
    assert public["strategy"]["factor_weights"]["technical_score"] == 0.30
    assert public["strategy"]["capital_mode"] == "manual"
    assert public["strategy"]["allocation_used_for_selection"] is False
    assert "allocation" not in public["strategy"]


def test_strategy_config_uses_top10_defaults():
    assert settings.strategy.stable_top_n == 10
    assert settings.strategy.aggressive_top_n == 10


def test_web_v8_artifact_defaults_to_versioned_feature_contract():
    assert settings.research.v8_artifact_path == "reports/ml_lab/v11_dual_artifact_feature_v2/artifact.json"


def test_strategy_weights_are_modern_fractions():
    total = sum(settings.strategy.factor_weights.values())
    assert 0.9 < total < 1.1


def test_historical_config_files_stay_legacy():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    legacy = (root / "config" / "dual_strategy_config.py").read_text(encoding="utf-8")
    assert "历史双策略配置文件" in legacy
    assert "当前主链路已迁移到 `src/quant_trading/config.py`" in legacy
    position_legacy = (root / "config" / "position_config.py").read_text(encoding="utf-8")
    decision_legacy = (root / "config" / "decision_config.py").read_text(encoding="utf-8")
    assert "历史持仓管理配置" in position_legacy
    assert "不得被 Web / Service 新代码作为资金分配依据" in position_legacy
    assert "不得被 Web / Service 新代码" in decision_legacy
