"""
AI分析配置

DeepSeek/语义分析路径在当前重建阶段暂缓。
不要在本文件中硬编码任何密钥；如未来恢复该模块，只允许从 .env 读取。
"""

import os

AI_ANALYSIS_ENABLED = os.getenv("AI_ANALYSIS_ENABLED", "false").lower() == "true"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 交易参数配置
TRADING_CONFIG = {
    # 止损止盈
    "stop_loss_pct": -3.5,      # 止损百分比
    "take_profit_1": 4.0,       # 第一止盈目标
    "take_profit_2": 7.0,       # 第二止盈目标
    
    # 仓位管理
    "max_position": 8.0,        # 单只股票最大仓位%
    "initial_position": 5.0,    # 首次买入仓位%
    "add_position": 3.0,        # 预留加仓仓位%
    
    # 交易时段（24小时制）
    "trading_sessions": [
        {"name": "早盘", "start": "09:30", "end": "10:00"},
        {"name": "午盘", "start": "13:00", "end": "13:30"},
        {"name": "尾盘", "start": "14:30", "end": "15:00"}
    ],
    
    # 加仓条件
    "add_position_trigger": 2.0,  # 上涨2%触发加仓
}

# AI分析提示词模板（增强版 - 包含完整历史数据）
ANALYSIS_PROMPT_TEMPLATE = """你是一位专业的A股超短线交易分析师，擅长结合量化指标、资金流数据和历史走势进行精准分析。

请根据以下股票数据，生成一份针对超短线交易者的分析报告：

【股票信息】
- 股票代码：{ts_code}
- 股票名称：{stock_name}
- 当前价格：{current_price:.2f}元

【量化评分】
- 综合评分：{score:.2f}分（满分100）
- 技术评分：{technical_score:.2f}分
- 北向资金评分：{north_bound_score:.2f}分
- 主力资金评分：{main_capital_score:.2f}分
- 龙虎榜评分：{dragon_tiger_score:.2f}分
- 板块强度：{sector_strength:.2f}分
- ATR波动率：{atr_volatility:.2f}分

【基本面数据】
- 市值：{market_cap:.2f}亿元
- ROE：{roe:.2f}%
- PE：{pe:.2f}

【资金流数据】
- 北向资金净流入：{north_flow:.2f}万元
- 主力资金净流入：{main_flow:.2f}万元
- 龙虎榜活跃度：{dragon_tiger_active:.2f}

【技术指标】
- MA5：{ma5:.2f}元，MA10：{ma10:.2f}元，MA20：{ma20:.2f}元
- MACD：{macd:.2f}，信号线：{signal:.2f}
- RSI：{rsi:.2f}

【历史走势分析（最近{history_days}天）】
{history_summary}

【交易习惯】
- 交易时段：早盘9:30-10:00、午盘13:00-13:30、尾盘14:30-15:00
- 止损：-3.5%
- 止盈：第一目标+4%，第二目标+7%
- 最大仓位：8%

请按以下格式输出分析报告（严格遵守格式，不要添加额外的markdown标记）：

【AI综合评价】
（80-120字，重点解读：
1. 资金流三因子（北向、主力、龙虎榜）的协同性及对短期走势的影响
2. 结合历史走势，分析当前位置的安全性和上涨空间
3. 技术面与资金面的共振情况）

【仓位建议】
建议仓位：X-X%
首次买入：X%
预留加仓：X%

【买入策略】
✅ 最佳时机：
  - 早盘9:30-10:00：（结合历史成交量特征，给出具体条件）
  - 下午13:00-13:30：（结合历史价格走势，给出具体条件）
  - 尾盘14:30-15:00：（结合历史资金流特征，给出具体条件）

💰 买入价位：
  - 激进：{aggressive_price:.2f}元（条件：开盘后快速突破，成交量放大）
  - 稳健：{stable_price:.2f}元（条件：回踩支撑位企稳）
  - 加仓价：{add_price:.2f}元（条件：上涨2%且资金持续流入）

【卖出策略】
❌ 止损价位：{stop_loss_price:.2f}元（-3.5%）
  - 跌破此价立即止损，不要犹豫

📈 止盈价位：
  - 第一目标：{take_profit_1_price:.2f}元（+4%），减仓50%
  - 第二目标：{take_profit_2_price:.2f}元（+7%），清仓

🔄 加仓条件：
  - 买入后上涨2%且成交量较前30分钟均值放大50%以上
  - 主力资金和北向资金同步流入

【观察要点】
🔍 早盘（9:30-10:00）观察：
  - 开盘5分钟成交量是否超过历史均值（历史均值：{avg_volume:.0f}手）
  - 主力资金流向（历史{history_days}天平均：{avg_main_flow:.2f}万元/天）
  - 价格是否站稳MA5（{ma5:.2f}元）

🔍 午盘（13:00-13:30）观察：
  - 是否回踩早盘高点不破
  - 主力资金是否护盘（观察大单托盘）
  - 成交量是否温和缩量

🔍 尾盘（14:30-15:00）观察：
  - 是否出现尾盘拉升
  - 资金流入是否加速（北向+主力）
  - 是否收在全天高位

【风险提示】
⚠️ 若出现以下情况立即离场：
  - 跌破止损位{stop_loss_price:.2f}元
  - 主力资金单日流出超过{risk_outflow:.2f}万元
  - 跌破MA20（{ma20:.2f}元）且无法快速收回
  - 成交量异常萎缩（低于历史均值50%）

【操作纪律】
✓ 严格执行止损，不抱幻想
✓ 只在三个时段交易，其他时间不操作
✓ 盈利达到目标及时止盈
✓ 不追高，不抄底

注意事项：
1. 所有价格计算基于当前价{current_price:.2f}元
2. 充分利用历史数据分析趋势和资金流特征
3. 重点关注资金流三因子的协同性
4. 结合历史成交量判断当前量能是否健康
5. 语言简洁专业，直击要点
"""

# API调用配置
API_CONFIG = {
    "temperature": 0.7,      # 创造性（0-1，越高越创造）
    "max_tokens": 2000,      # 最大输出长度
    "timeout": 30,           # 超时时间（秒）
}
