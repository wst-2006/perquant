"""
历史决策引擎配置。

主链路当前不再以此文件作为运行时权威配置。文件内的固定
`buy_amount` 仅是旧版叙述残留，不得被 Web / Service 新代码
作为建议买入金额、目标仓位或自动交易依据。
"""

# ============================================
# 早盘决策配置
# ============================================
MORNING_DECISION_CONFIG = {
    'enabled': True,
    'candidate_pool_size': 20,  # 候选池大小
    'final_signal_count': 3,    # 最终信号数量

    # 集合竞价因子权重（暂不实现，预留）
    'auction_score_weights': {
        'change_pct': 0.3,      # 涨跌幅
        'volume_ratio': 0.3,    # 量比
        'buy_sell_ratio': 0.4   # 委买委卖比
    },

    # 集合竞价过滤条件（暂不实现，预留）
    'auction_filters': {
        'max_change_pct': 7.0,        # 最大涨幅7%
        'min_volume_ratio': 1.5,      # 最小量比1.5
        'min_buy_sell_ratio': 1.2     # 最小委买委卖比1.2
    },

    # 买入金额配置
    'buy_amount': 8000  # 进攻型每只8000元
}

# ============================================
# 尾盘决策配置
# ============================================
AFTERNOON_DECISION_CONFIG = {
    'enabled': True,
    'candidate_pool_size': 30,  # 候选池大小
    'final_signal_count': 4,    # 最终信号数量

    # 盘中因子权重（暂不实现，预留）
    'intraday_score_weights': {
        'original_score': 0.7,   # 原始评分
        'intraday_flow': 0.2,    # 盘中资金流
        'intraday_trend': 0.1    # 盘中走势
    },

    # 盘中过滤条件（暂不实现，预留）
    'intraday_filters': {
        'max_intraday_change': 5.0,    # 当日涨幅<5%
        'min_sector_strength': -5.0    # 板块5日跌幅>-5%
    },

    # 买入金额配置
    'buy_amount': 10000  # 稳健型每只10000元
}

# ============================================
# 因子权重配置
# ============================================
FACTOR_WEIGHTS = {
    'aggressive': {
        'technical': 0.40,      # 技术面
        'capital': 0.35,        # 资金面
        'fundamental': 0.15,    # 基本面
        'auction': 0.10         # 集合竞价（暂不实现）
    },
    'stable': {
        'technical': 0.30,      # 技术面
        'capital': 0.20,        # 资金面
        'fundamental': 0.25,    # 基本面
        'market': 0.15,         # 市场环境
        'sector': 0.10          # 板块强度
    }
}

# ============================================
# 前置风控配置
# ============================================
PRE_RISK_CONTROL = {
    'enabled': True,
    'max_expected_drawdown': 0.10,      # 最大预期回撤10%
    'min_daily_amount': 50000000,       # 最小日均成交额5000万元
    'max_goodwill_ratio': 0.30,         # 最大商誉占比30%
    'max_debt_ratio': 0.70,             # 最大资产负债率70%
    'lookback_days': 60                 # 回溯天数60天
}

# ============================================
# 决策卡片配置
# ============================================
DECISION_CARD_CONFIG = {
    'target_profit_pct': 0.05,  # 目标盈利5%
    'stop_loss_pct': -0.08,     # 止损-8%

    # 置信度星级阈值
    'confidence_thresholds': {
        5: 90,  # ≥90分 → 5星
        4: 80,  # ≥80分 → 4星
        3: 70,  # ≥70分 → 3星
        2: 60,  # ≥60分 → 2星
        1: 0    # <60分 → 1星
    }
}

# ============================================
# Tushare字段映射配置
# ============================================
TUSHARE_FIELD_MAPPING = {
    'moneyflow': {
        'net_mf_amount': ['net_mf_amount', 'net_mf_amt'],  # 主力净流入金额
        'net_mf_vol': ['net_mf_vol', 'net_mf_volume']      # 主力净流入量
    },
    'daily': {
        'amount': ['amount', 'amt'],  # 成交额（千元）
        'volume': ['vol', 'volume']   # 成交量（手）
    },
    'balancesheet': {
        'goodwill': ['goodwill'],  # 商誉
        'total_equity': ['total_hldr_eqy_exc_min_int', 'total_equity'],  # 股东权益合计
        'total_liab': ['total_liab', 'total_liabilities'],  # 负债合计
        'total_assets': ['total_assets', 'total_asset']  # 资产总计
    }
}
