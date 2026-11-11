from src.quant_trading.analysis.factors import FactorEngine
from src.quant_trading.analysis.market import MarketAnalyzer


def test_market_advice_is_not_position_allocation():
    advice = MarketAnalyzer._trading_advice("red")
    assert advice["candidate_status"] == "watch_only"
    assert "short_term_core" not in advice
    assert "cash" not in advice


def test_factor_explains_candidates_without_capital_fields():
    scores = {
        "technical_score": 80,
        "capital_score": None,
        "sector_strength": 75,
        "valuation_score": 45,
        "growth_score": 70,
        "quality_score": 55,
    }
    reasons = FactorEngine._selection_reasons(scores, {"signal": "green"}, 75)
    risks = FactorEngine._risk_tags(scores, {"valid": True}, {"valid": False}, {"signal": "green"})
    profile = FactorEngine._trade_profile("aggressive", "red")

    assert "技术强度较高" in reasons
    assert "资金流缺失" in risks
    assert profile["style"] == "进攻候选"
    assert profile["market_gate"] == "watch_only"
    assert "buy_amount" not in profile
