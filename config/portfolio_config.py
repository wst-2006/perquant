"""
投资组合配置
定义70%稳健+30%进攻双策略的所有参数
"""

# ============================================
# 投资组合基础配置
# ============================================
PORTFOLIO_CONFIG = {
    'total_capital': 1000000,  # 总资金（元）
    'stable_target': 0.70,     # 稳健型目标配置（70%）
    'aggressive_target': 0.30, # 进攻型目标配置（30%）
    'rebalance_tolerance': 0.05, # 再平衡容忍度（±5%）

    # 稳健型仓位规则
    'stable_position_rules': {
        'min_positions': 5,          # 最少5个仓位
        'max_positions': 8,          # 最多8个仓位
        'min_position_pct': 0.08,    # 单仓最小8%
        'max_position_pct': 0.15,    # 单仓最大15%
        'target_hold_days': (30, 90) # 目标持有30-90天
    },

    # 进攻型仓位规则
    'aggressive_position_rules': {
        'min_positions': 8,           # 最少8个仓位
        'max_positions': 12,          # 最多12个仓位
        'min_position_pct': 0.025,    # 单仓最小2.5%
        'max_position_pct': 0.0375,   # 单仓最大3.75%
        'target_hold_days': (5, 20)   # 目标持有5-20天
    }
}

# ============================================
# 策略分类标准
# ============================================
STRATEGY_CLASSIFICATION = {
    # 稳健型标准
    'stable_criteria': {
        'market_cap_min': 100000000000,  # 市值>1000亿（单位：元）
        'pe_max': 30,                     # PE<30
        'roe_min': 0.10,                  # ROE>10%
        'volatility_max': 0.30,           # 年化波动率<30%
        'dividend_yield_min': 0.02,       # 股息率>2%
        'tech_score_min': 60,             # 技术评分>60
        'weight': {
            'market_cap': 0.20,
            'valuation': 0.25,
            'profitability': 0.25,
            'stability': 0.20,
            'dividend': 0.10
        }
    },

    # 进攻型标准
    'aggressive_criteria': {
        'market_cap_range': (20000000000, 200000000000),  # 市值200亿-2000亿
        'growth_rate_min': 0.20,          # 增长率>20%
        'momentum_score_min': 70,         # 动量评分>70
        'volume_surge_min': 1.5,          # 成交量放大>1.5倍
        'tech_score_min': 70,             # 技术评分>70
        'weight': {
            'growth': 0.30,
            'momentum': 0.30,
            'volume': 0.20,
            'technical': 0.20
        }
    }
}

# ============================================
# 仓位计算配置
# ============================================
POSITION_SIZING = {
    # 稳健型仓位计算
    'stable': {
        'base_pct': 0.08,      # 基础仓位8%
        'max_pct': 0.15,       # 最大仓位15%
        'score_threshold': 60, # 评分阈值
        'score_range': 40,     # 评分范围（60-100）
        'formula': 'base_pct + (score - score_threshold) / score_range * (max_pct - base_pct)'
    },

    # 进攻型仓位计算
    'aggressive': {
        'base_pct': 0.025,     # 基础仓位2.5%
        'max_pct': 0.0375,     # 最大仓位3.75%
        'score_threshold': 60, # 评分阈值
        'score_range': 40,     # 评分范围（60-100）
        'formula': 'base_pct + (score - score_threshold) / score_range * (max_pct - base_pct)'
    }
}

# ============================================
# 再平衡规则
# ============================================
REBALANCE_RULES = {
    # 触发再平衡的条件
    'triggers': {
        'allocation_deviation': 0.05,  # 配置偏离>5%
        'position_score_min': 50,      # 持仓评分<50
        'hold_days_max_stable': 90,    # 稳健型持有>90天
        'hold_days_max_aggressive': 20, # 进攻型持有>20天
        'loss_threshold': -0.08        # 亏损>8%
    },

    # 再平衡操作
    'actions': {
        'sell_score_threshold': 50,    # 评分<50卖出
        'reduce_overweight': 0.03,     # 超配>3%减仓
        'increase_underweight': 0.03   # 低配>3%加仓
    }
}

# ============================================
# 风险控制配置
# ============================================
RISK_CONTROL = {
    # 单票风险
    'single_position': {
        'max_loss_pct': 0.10,          # 单票最大回撤10%
        'stop_loss_pct': 0.08,         # 止损线8%
        'take_profit_pct': 0.05        # 止盈线5%（动态）
    },

    # 组合风险
    'portfolio': {
        'max_sector_concentration': 0.30,  # 单板块最大30%
        'max_correlation': 0.70,           # 最大相关性0.7
        'min_diversification': 10          # 最少10个仓位
    },

    # 市场风险
    'market': {
        'min_market_score': 40,        # 市场评分<40停止买入
        'bear_market_threshold': 30,   # 熊市阈值
        'bull_market_threshold': 70    # 牛市阈值
    }
}

# ============================================
# 报告配置
# ============================================
REPORT_CONFIG = {
    'output_dir': 'reports/portfolio',
    'formats': ['excel', 'html', 'json'],

    # 报告章节
    'sections': {
        'overview': True,              # 投资组合概览
        'strategy_performance': True,  # 策略表现
        'position_analysis': True,     # 持仓分析
        'risk_metrics': True,          # 风险指标
        'optimization': True,          # 优化建议
        'charts': True                 # 图表
    },

    # 性能指标
    'metrics': [
        'total_return',        # 总收益率
        'annualized_return',   # 年化收益率
        'sharpe_ratio',        # 夏普比率
        'max_drawdown',        # 最大回撤
        'win_rate',            # 胜率
        'profit_loss_ratio',   # 盈亏比
        'turnover_rate'        # 换手率
    ]
}

# ============================================
# CSV导入配置
# ============================================
CSV_IMPORT_CONFIG = {
    # 必需字段
    'required_fields': ['ts_code', 'name', 'cost', 'quantity'],

    # 可选字段
    'optional_fields': ['buy_date', 'strategy', 'sector', 'notes'],

    # 字段映射（支持不同的CSV格式）
    'field_mapping': {
        'ts_code': ['ts_code', 'code', '股票代码', '代码'],
        'name': ['name', 'stock_name', '股票名称', '名称'],
        'cost': ['cost', 'buy_price', 'price', '成本价', '买入价'],
        'quantity': ['quantity', 'amount', 'shares', '数量', '股数'],
        'buy_date': ['buy_date', 'date', 'trade_date', '买入日期', '日期'],
        'strategy': ['strategy', 'type', '策略', '类型'],
        'sector': ['sector', 'industry', '板块', '行业'],
        'notes': ['notes', 'memo', 'remark', '备注', '说明']
    },

    # 数据验证
    'validation': {
        'ts_code_pattern': r'^\d{6}\.(SH|SZ)$',  # 股票代码格式
        'cost_min': 0.01,                         # 最小成本价
        'cost_max': 10000,                        # 最大成本价
        'quantity_min': 100,                      # 最小数量（1手）
        'quantity_multiple': 100                  # 数量必须是100的倍数
    }
}


if __name__ == '__main__':
    # 测试配置
    print("投资组合配置测试")
    print("="*60)

    print(f"\n总资金: {PORTFOLIO_CONFIG['total_capital']:,}元")
    print(f"稳健型目标: {PORTFOLIO_CONFIG['stable_target']:.0%}")
    print(f"进攻型目标: {PORTFOLIO_CONFIG['aggressive_target']:.0%}")

    stable_capital = PORTFOLIO_CONFIG['total_capital'] * PORTFOLIO_CONFIG['stable_target']
    aggressive_capital = PORTFOLIO_CONFIG['total_capital'] * PORTFOLIO_CONFIG['aggressive_target']

    print(f"\n稳健型资金池: {stable_capital:,}元")
    print(f"  - 仓位数: {PORTFOLIO_CONFIG['stable_position_rules']['min_positions']}-{PORTFOLIO_CONFIG['stable_position_rules']['max_positions']}个")
    print(f"  - 单仓范围: {stable_capital * PORTFOLIO_CONFIG['stable_position_rules']['min_position_pct']:,.0f}-{stable_capital * PORTFOLIO_CONFIG['stable_position_rules']['max_position_pct']:,.0f}元")

    print(f"\n进攻型资金池: {aggressive_capital:,}元")
    print(f"  - 仓位数: {PORTFOLIO_CONFIG['aggressive_position_rules']['min_positions']}-{PORTFOLIO_CONFIG['aggressive_position_rules']['max_positions']}个")
    print(f"  - 单仓范围: {aggressive_capital * PORTFOLIO_CONFIG['aggressive_position_rules']['min_position_pct']:,.0f}-{aggressive_capital * PORTFOLIO_CONFIG['aggressive_position_rules']['max_position_pct']:,.0f}元")

    print("\n✅ 配置加载成功")
