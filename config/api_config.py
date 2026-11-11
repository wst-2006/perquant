"""
API配置文件
从环境变量加载敏感信息

使用方法:
1. 复制 .env.example 为 .env
2. 在 .env 中填入你的真实凭证
3. .env 文件不会被提交到git
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# ============================================
# Tushare Pro 配置
# ============================================
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
TUSHARE_POINTS = int(os.getenv('TUSHARE_POINTS', '2200'))

# ============================================
# 数据库配置
# ============================================
DB_CONFIG = {
    "host": os.getenv('DB_HOST', 'localhost'),
    "port": int(os.getenv('DB_PORT', '5432')),
    "database": os.getenv('DB_NAME', 'quant_db'),
    "user": os.getenv('DB_USER', 'postgres'),
    "password": os.getenv('DB_PASSWORD')
}

# SQLAlchemy连接字符串
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# ============================================
# 配置验证
# ============================================
def validate_config():
    """验证配置是否完整"""
    errors = []

    if not TUSHARE_TOKEN:
        errors.append("TUSHARE_TOKEN 未设置")

    if not DB_CONFIG['password']:
        errors.append("DB_PASSWORD 未设置")

    if errors:
        raise ValueError(f"配置错误:\n" + "\n".join(f"  - {e}" for e in errors))

    return True

# 启动时验证配置
try:
    validate_config()
except ValueError as e:
    print(f"⚠️ {e}")
    print("\n请按照以下步骤配置:")
    print("1. 复制 .env.example 为 .env")
    print("2. 在 .env 中填入你的真实凭证")
    print("3. 重新运行程序")

# ============================================
# 缓存配置
# ============================================
CACHE_CONFIG = {
    "enabled": True,
    "cache_dir": "data/cache",
    "expire_hours": 24  # 缓存过期时间（小时）
}

# ============================================
# 日志配置
# ============================================
LOG_CONFIG = {
    "log_dir": "data/logs",
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "rotation": "1 day",  # 日志轮转周期
    "retention": "30 days"  # 日志保留时间
}

# ============================================
# API请求配置
# ============================================
REQUEST_CONFIG = {
    "timeout": 30,  # 请求超时时间（秒）
    "retry_times": 3,  # 重试次数
    "retry_delay": 1,  # 重试延迟（秒）
}

# ============================================
# 数据更新配置
# ============================================
UPDATE_CONFIG = {
    "history_years": 3,  # 下载历史数据年数
    "daily_update_time": "18:00",  # 每日更新时间
    "realtime_interval": 3,  # 实时数据更新间隔（秒）
}

