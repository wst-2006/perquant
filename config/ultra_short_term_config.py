"""
超短线策略配置

根据用户偏好：
- 风险暴露：4%（标准型）
- 调仓频率：3天
- 选股侧重：技术+资金优先
- 市场适配：手动调整
"""

# 超短线策略配置
ULTRA_SHORT_TERM_CONFIG = {
    # 选股策略
    'selection': {
        'strategy_type': 'ultra_short_term',  # 超短线策略
        'stock_pool': 'all',  # 全市场A股（约5484只）
        'top_n': 20,  # 选出20只
        'rebalance_days': 3,  # 3天调仓一次
        'min_score': 50,  # 最低综合评分50分（降低标准，关注排名）

        # 股票池选项说明：
        # 'all' - 全市场A股（约5484只，推荐用于超短线和进攻型策略）
        # 'hs300' - 沪深300（300只，推荐用于稳健型策略）
        # 'zz500' - 中证500（500只）
        # 'hs300_zz500' - 沪深300+中证500（约800只）

        # 因子权重（技术+资金优先）
        'factor_weights': {
            'technical_score': 35,      # 技术面35%
            'atr_volatility': 5,        # 波动率5%
            'north_bound_score': 20,    # 北向资金20%
            'main_capital_flow': 10,    # 主力资金10%
            'dragon_tiger_activity': 5, # 龙虎榜5%
            'sector_strength': 15,      # 板块强度15%
            'market_cap': 5,            # 市值5%
            'roe': 3,                   # ROE 3%
            'pe_percentile': 2,         # PE百分位2%
        }
    },

    # 仓位管理
    'position': {
        'method': 'atr_volatility',  # ATR波动性百分比
        'risk_percent': 4,  # 每次调仓最大损失4%
        'atr_period': 14,  # ATR周期14天
        'atr_multiplier': 2,  # 止损距离=2倍ATR
        'max_single_position': 15,  # 单只最大15%
        'max_total_position': 95,  # 总仓位最大95%

        # 超短线特殊规则
        'quick_exit': {
            'enabled': True,
            'max_hold_days': 5,         # 最大持仓5天
            'profit_target': 8,         # 盈利目标8%
            'stop_loss': -4,            # 止损-4%
        }
    },

    # 风控参数
    'risk': {
        'stop_loss_method': 'atr_dynamic',  # ATR动态止损
        'max_hold_days': 5,  # 最大持仓5天
        'max_daily_loss': -3,  # 单日最大亏损-3%
        'max_weekly_loss': -8,  # 单周最大亏损-8%
        'max_drawdown': -10,  # 最大回撤-10%

        # 超短线快速止盈
        'quick_profit': {
            'enabled': True,
            'days': 2,              # 持仓≥2天
            'profit': 8,            # 盈利≥8%
            'action': 'reduce',     # 减仓50%
            'ratio': 0.5
        }
    },

    # 市场环境分析（手动调整模式）
    'market_adapt': {
        'mode': 'manual',  # 手动调整模式
        'analysis_enabled': True,  # 启用市场分析
        'alert_threshold': 20,  # 市场评分变化>20分时提醒

        # 系统会提供建议，但不自动执行
        'suggestions': {
            'risk_percent_range': [2, 6],  # 建议风险暴露范围
            'rebalance_days_range': [3, 10],  # 建议调仓频率范围
            'position_limit_range': [50, 95],  # 建议仓位上限范围
        }
    }
}


if __name__ == '__main__':
    print("超短线策略配置:")
    print("=" * 60)

    print("\n选股配置:")
    for k, v in ULTRA_SHORT_TERM_CONFIG['selection'].items():
        print(f"  {k}: {v}")

    print("\n仓位管理:")
    for k, v in ULTRA_SHORT_TERM_CONFIG['position'].items():
        print(f"  {k}: {v}")

    print("\n风控参数:")
    for k, v in ULTRA_SHORT_TERM_CONFIG['risk'].items():
        print(f"  {k}: {v}")
