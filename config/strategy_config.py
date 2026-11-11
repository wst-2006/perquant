"""
历史策略参数配置。

当前主链路以 `src/quant_trading/config.py` 为准。
"""

# ============================================
# 小资金优势策略配置
# ============================================

# 仓位分配
POSITION_CONFIG = {
    "total_capital": 1000000,  # 总资金（元）
    
    # 仓位分配比例
    "short_term_ratio": 0.60,  # 超短线仓位 60%
    "swing_ratio": 0.30,       # 波段仓位 30%
    "aggressive_ratio": 0.10,  # 激进仓位 10%
    
    # 单股最大仓位
    "max_single_position": 0.30,  # 单股最大30%
    
    # 现金储备
    "cash_reserve_ratio": 0.10,  # 保留10%现金
}

# ============================================
# 超短线策略（1-3天）
# ============================================
SHORT_TERM_STRATEGY = {
    "name": "情绪周期+龙头战法",
    "holding_days": (1, 3),
    "target_return": 0.05,  # 目标收益5%
    "stop_loss": -0.03,     # 止损-3%
    
    # 选股条件
    "filters": {
        # 技术面
        "ma_alignment": True,  # MA5 > MA10 > MA20
        "macd_golden_cross_days": 3,  # MACD金叉3日内
        "rsi_range": (50, 70),  # RSI强势区间
        "volume_ratio_min": 1.5,  # 量比 > 1.5
        "price_change_range": (0, 0.07),  # 涨幅0-7%
        
        # 资金面
        "main_inflow": True,  # 主力净流入
        "continuous_inflow_days": 1,  # 连续流入天数
        
        # 市场面
        "relative_strength_min": 1.2,  # 相对大盘强度 > 1.2
        "sector_rank_percentile": 0.3,  # 板块排名前30%
        
        # 基本面过滤
        "market_cap_range": (5e9, 50e9),  # 市值50亿-500亿
        "exclude_st": True,  # 排除ST
    },
    
    # 买入时机
    "entry_signals": {
        "call_auction_volume_ratio": 3.0,  # 竞价量比 > 3
        "first_30min_strength": True,  # 开盘30分钟强势
        "above_avg_price": 0.7,  # 70%时间在均价线上
    },
    
    # 卖出时机
    "exit_signals": {
        "profit_target": 0.05,  # 盈利5%
        "stop_loss": -0.03,  # 止损-3%
        "sentiment_peak": True,  # 情绪高潮
        "volume_shrink": True,  # 缩量
    }
}

# ============================================
# 波段策略（5-15天）
# ============================================
SWING_STRATEGY = {
    "name": "突破回踩+趋势持有",
    "holding_days": (5, 15),
    "target_return": 0.15,  # 目标收益15%
    "stop_loss": -0.05,     # 止损-5%
    
    # 选股条件
    "filters": {
        # 技术面
        "above_ma20": True,  # 价格在20日线上
        "macd_divergence": True,  # MACD底背离
        "support_level": True,  # 在支撑位附近
        "volume_moderate": True,  # 成交量温和放大
        
        # 资金面
        "main_inflow_5d": True,  # 5日主力净流入
        "institutional_buying": True,  # 机构买入
        
        # 市场面
        "relative_strength_min": 1.3,  # 相对强度 > 1.3
        "sector_uptrend": True,  # 板块上升期
        
        # 基本面
        "roe_min": 0.10,  # ROE > 10%
        "revenue_growth_min": 0.10,  # 营收增长 > 10%
        "catalyst": True,  # 有催化剂（财报/政策）
    },
    
    # 买入时机
    "entry_signals": {
        "pullback_to_support": True,  # 回踩支撑位
        "volume_confirmation": True,  # 量能确认
        "macd_golden_cross": True,  # MACD金叉
    },
    
    # 卖出时机
    "exit_signals": {
        "profit_target": 0.15,  # 盈利15%
        "stop_loss": -0.05,  # 止损-5%
        "break_ma20": True,  # 跌破20日线
        "volume_peak": True,  # 放量滞涨
    }
}

# ============================================
# 激进策略（打板/超短）
# ============================================
AGGRESSIVE_STRATEGY = {
    "name": "首板战法",
    "holding_days": (1, 2),
    "target_return": 0.10,  # 目标收益10%
    "stop_loss": -0.05,     # 止损-5%
    
    # 选股条件
    "filters": {
        "first_limit_up": True,  # 首板
        "limit_up_time": "14:00",  # 封板时间早于14:00
        "seal_strength": 0.8,  # 封单强度 > 80%
        "turnover_rate_range": (0.05, 0.20),  # 换手率5-20%
        "sector_hot": True,  # 热门板块
        "news_catalyst": True,  # 有新闻催化
    },
    
    # 买入时机
    "entry_signals": {
        "limit_up_sealed": True,  # 涨停封死
        "large_buy_orders": True,  # 大单封单
    },
    
    # 卖出时机
    "exit_signals": {
        "next_day_open": "high",  # 次日高开卖出
        "profit_target": 0.10,
        "stop_loss": -0.05,
    }
}

# ============================================
# 技术指标参数
# ============================================
INDICATOR_PARAMS = {
    # 均线参数
    "ma_periods": [5, 10, 20, 60, 120, 250],
    
    # MACD参数
    "macd": {
        "fast": 12,
        "slow": 26,
        "signal": 9
    },
    
    # RSI参数
    "rsi_period": 14,
    
    # KDJ参数
    "kdj": {
        "n": 9,
        "m1": 3,
        "m2": 3
    },
    
    # 布林带参数
    "bollinger": {
        "period": 20,
        "std": 2
    },
    
    # ATR参数
    "atr_period": 14,
}

# ============================================
# 风险控制参数
# ============================================
RISK_CONTROL = {
    # 单日最大亏损
    "max_daily_loss": -0.03,  # -3%
    
    # 单周最大亏损
    "max_weekly_loss": -0.08,  # -8%
    
    # 单月最大亏损
    "max_monthly_loss": -0.15,  # -15%
    
    # 连续亏损次数限制
    "max_consecutive_losses": 3,
    
    # 单股止损
    "stock_stop_loss": -0.05,  # -5%
    
    # 单股止盈
    "stock_take_profit": 0.20,  # 20%
    
    # 回撤控制
    "max_drawdown": -0.20,  # 最大回撤-20%
}

# ============================================
# 市场环境判断参数
# ============================================
MARKET_ENVIRONMENT = {
    # 牛市
    "bull_market": {
        "position_ratio": 0.90,  # 90%仓位
        "aggressive_ratio": 0.15,  # 15%激进仓位
    },
    
    # 震荡市
    "sideways_market": {
        "position_ratio": 0.70,  # 70%仓位
        "aggressive_ratio": 0.10,  # 10%激进仓位
    },
    
    # 熊市
    "bear_market": {
        "position_ratio": 0.30,  # 30%仓位
        "aggressive_ratio": 0.00,  # 0%激进仓位
    },
}

# ============================================
# 板块轮动参数
# ============================================
SECTOR_ROTATION = {
    # 板块强度阈值
    "strong_sector_threshold": 0.03,  # 涨幅 > 3%
    
    # 资金流入阈值
    "inflow_threshold": 1e8,  # 1亿
    
    # 持续天数
    "duration_days": 3,
    
    # 龙头股判断
    "leader_criteria": {
        "sector_rank": 3,  # 板块内前3名
        "relative_strength": 1.5,  # 相对板块强度 > 1.5
        "volume_ratio": 2.0,  # 量比 > 2.0
    }
}



