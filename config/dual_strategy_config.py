"""
历史双策略配置文件。

当前主链路已迁移到 `src/quant_trading/config.py`，这里仅保留历史参考。

注意：
1. 不再作为主配置入口
2. 不再作为 Web / Service 的权威来源
3. 仅用于回看旧实现
4. 文件内百分比权重沿用旧叙述，不代表当前运行口径
"""

import json
import os

# 运行时配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'strategy_runtime_config.json')

# ============================================
# 默认配置：稳健型策略
# ============================================
DEFAULT_STABLE_CONFIG = {
    'name': '稳健型策略',
    'type': 'stable',
    'top_n': 20,
    'min_score': 50,
    'rebalance_days': 5,  # 5天调仓
    
    # 因子权重（总和=100%）
    'factor_weights': {
        'technical_score': 25,          # 技术综合评分 25%
        'atr_volatility': 5,            # ATR波动率 5%
        'north_bound_score': 15,        # 北向资金 15%
        'main_capital_score': 20,       # 主力资金 20%
        'dragon_tiger_score': 5,        # 龙虎榜 5%
        'sector_strength': 15,          # 板块强度 15%
        'market_cap': 3,                # 市值 3%
        'roe': 5,                       # ROE 5%
        'pe_percentile': 2,             # PE百分位 2%
    },
    
    # 数据周期
    'data_period': {
        'technical_days': 60,           # 技术指标60天
        'capital_flow_days': 4,         # 资金流4天
        'capital_flow_weights': {
            'daily': 0.4,               # 当日40%
            'multi_day': 0.6            # 三日60%
        },
        'preliminary_filter_top_n': 1000  # 初步筛选1000只
    },
    
    # 描述
    'description': '稳健型策略：注重资金持续流入和中期趋势，适合5天调仓周期'
}

# ============================================
# 默认配置：进攻型策略
# ============================================
DEFAULT_AGGRESSIVE_CONFIG = {
    'name': '进攻型策略',
    'type': 'aggressive',
    'top_n': 20,
    'min_score': 50,
    'rebalance_days': 3,  # 3天调仓
    
    # 因子权重（总和=100%）
    'factor_weights': {
        'technical_score': 40,          # 技术综合评分 40%
        'atr_volatility': 5,            # ATR波动率 5%
        'north_bound_score': 12,        # 北向资金 12%
        'main_capital_score': 18,       # 主力资金 18%
        'dragon_tiger_score': 5,        # 龙虎榜 5%
        'sector_strength': 15,          # 板块强度 15%
        'market_cap': 3,                # 市值 3%
        'roe': 0,                       # ROE 0%（忽略盈利能力）
        'pe_percentile': 2,             # PE百分位 2%
    },
    
    # 数据周期
    'data_period': {
        'technical_days': 20,           # 技术指标20天
        'capital_flow_days': 2,         # 资金流2天
        'capital_flow_weights': {
            'daily': 0.6,               # 当日60%
            'multi_day': 0.4            # 昨日40%
        },
        'preliminary_filter_top_n': 2000  # 初步筛选2000只
    },
    
    # 描述
    'description': '进攻型策略：捕捉短期技术信号和资金热点，适合3天调仓周期'
}

# ============================================
# 配置管理函数
# ============================================

def load_config(strategy_type: str) -> dict:
    """
    加载配置（优先从运行时配置文件读取）
    
    Args:
        strategy_type: 策略类型 'stable' 或 'aggressive'
        
    Returns:
        配置字典
    """
    # 1. 尝试从运行时配置文件读取
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                runtime_config = json.load(f)
                if strategy_type in runtime_config:
                    return runtime_config[strategy_type]
        except Exception as e:
            print(f"读取运行时配置失败: {e}，使用默认配置")
    
    # 2. 返回默认配置
    if strategy_type == 'stable':
        return DEFAULT_STABLE_CONFIG.copy()
    elif strategy_type == 'aggressive':
        return DEFAULT_AGGRESSIVE_CONFIG.copy()
    else:
        raise ValueError(f"未知的策略类型: {strategy_type}")

def save_config(strategy_type: str, config: dict):
    """
    保存配置到运行时配置文件
    
    Args:
        strategy_type: 策略类型 'stable' 或 'aggressive'
        config: 配置字典
    """
    # 读取现有配置
    runtime_config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                runtime_config = json.load(f)
        except:
            pass
    
    # 更新配置
    runtime_config[strategy_type] = config
    
    # 保存到文件
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(runtime_config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 配置已保存: {strategy_type}")

def reset_to_default(strategy_type: str):
    """
    恢复默认配置
    
    Args:
        strategy_type: 策略类型 'stable' 或 'aggressive'
    """
    if strategy_type == 'stable':
        config = DEFAULT_STABLE_CONFIG.copy()
    elif strategy_type == 'aggressive':
        config = DEFAULT_AGGRESSIVE_CONFIG.copy()
    else:
        raise ValueError(f"未知的策略类型: {strategy_type}")
    
    save_config(strategy_type, config)
    print(f"✅ 已恢复默认配置: {strategy_type}")

def validate_weights(config: dict) -> bool:
    """
    验证权重总和是否为100%
    
    Args:
        config: 配置字典
        
    Returns:
        True表示验证通过，False表示验证失败
    """
    if 'factor_weights' not in config:
        return False
    
    weights = config['factor_weights']
    total = sum(weights.values())
    
    # 允许0.1%的误差
    if abs(total - 100) > 0.1:
        print(f"❌ 权重总和验证失败: {total}% (应为100%)")
        return False
    
    return True

def get_all_configs() -> dict:
    """
    获取所有策略配置
    
    Returns:
        {'stable': {...}, 'aggressive': {...}}
    """
    return {
        'stable': load_config('stable'),
        'aggressive': load_config('aggressive')
    }

def print_config(strategy_type: str):
    """
    打印配置信息
    
    Args:
        strategy_type: 策略类型 'stable' 或 'aggressive'
    """
    config = load_config(strategy_type)
    
    print("="*60)
    print(f"策略配置: {config['name']}")
    print("="*60)
    print(f"类型: {config['type']}")
    print(f"选股数量: {config['top_n']}")
    print(f"最低评分: {config['min_score']}")
    print(f"调仓频率: {config['rebalance_days']}天")
    print(f"描述: {config.get('description', 'N/A')}")
    
    print("\n因子权重:")
    for factor, weight in config['factor_weights'].items():
        print(f"  {factor}: {weight}%")
    
    print(f"\n权重总和: {sum(config['factor_weights'].values())}%")
    
    print("\n数据周期:")
    for key, value in config['data_period'].items():
        print(f"  {key}: {value}")
    
    print("="*60)


if __name__ == '__main__':
    # 测试代码
    print("\n=== 测试配置管理 ===\n")
    
    # 1. 打印默认配置
    print("1. 稳健型策略配置:")
    print_config('stable')
    
    print("\n2. 进攻型策略配置:")
    print_config('aggressive')
    
    # 2. 验证权重
    print("\n3. 验证权重:")
    stable_config = load_config('stable')
    aggressive_config = load_config('aggressive')
    
    print(f"稳健型权重验证: {'✅ 通过' if validate_weights(stable_config) else '❌ 失败'}")
    print(f"进攻型权重验证: {'✅ 通过' if validate_weights(aggressive_config) else '❌ 失败'}")
    
    # 3. 测试保存和加载
    print("\n4. 测试保存配置:")
    test_config = stable_config.copy()
    test_config['top_n'] = 30  # 修改选股数量
    save_config('stable', test_config)
    
    loaded_config = load_config('stable')
    print(f"保存后加载的选股数量: {loaded_config['top_n']} (预期30)")
    
    # 4. 恢复默认
    print("\n5. 恢复默认配置:")
    reset_to_default('stable')
    
    restored_config = load_config('stable')
    print(f"恢复后的选股数量: {restored_config['top_n']} (预期20)")
