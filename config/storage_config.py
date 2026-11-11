"""
历史数据存储配置。

当前主链路优先使用数据库和 `reports/` 产物，旧路径仅作参考。
"""

import os
from datetime import datetime

# 数据根目录
DATA_ROOT = 'data'

# 数据存储路径模板
STORAGE_PATHS = {
    # 决策日志
    'decision_logs': os.path.join(DATA_ROOT, 'decision_logs', '{date}_decision.json'),

    # 执行记录
    'execution_logs': os.path.join(DATA_ROOT, 'execution_logs', '{date}_execution.json'),

    # 绩效归因
    'attribution_history': os.path.join(DATA_ROOT, 'attribution', 'weekly_attribution.json'),
    'factor_ic_history': os.path.join(DATA_ROOT, 'attribution', 'factor_ic_history.csv'),

    # 缓存数据
    'candidate_pool': os.path.join(DATA_ROOT, 'cache', 'candidate_pool_{date}.json'),
    'auction_data': os.path.join(DATA_ROOT, 'cache', 'auction_data_{date}.json'),
    'realtime_data': os.path.join(DATA_ROOT, 'cache', 'realtime_data_{date}.json'),

    # 周报
    'weekly_reports': os.path.join(DATA_ROOT, 'reports', 'weekly_report_{week}.md')
}

def get_storage_path(key, **kwargs):
    """
    获取存储路径（自动创建目录）

    Args:
        key: 路径键名
        **kwargs: 格式化参数（如date='20240101'）

    Returns:
        str: 完整的文件路径
    """
    path_template = STORAGE_PATHS.get(key)
    if not path_template:
        raise ValueError(f"未知的存储路径键: {key}")

    # 格式化路径
    path = path_template.format(**kwargs)

    # 自动创建目录
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    return path

def init_storage_dirs():
    """初始化所有存储目录"""
    for path_template in STORAGE_PATHS.values():
        # 提取目录部分（去掉占位符）
        dir_path = os.path.dirname(path_template.split('{')[0])
        os.makedirs(dir_path, exist_ok=True)

    print("✅ 数据存储目录初始化完成")

if __name__ == '__main__':
    # 测试：初始化所有目录
    init_storage_dirs()

    # 测试：获取路径
    test_date = datetime.now().strftime('%Y%m%d')
    decision_log_path = get_storage_path('decision_logs', date=test_date)
    print(f"决策日志路径: {decision_log_path}")
