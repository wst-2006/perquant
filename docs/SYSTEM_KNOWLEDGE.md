# 系统知识

本文档仅保留当前仓库真实在用的主链路说明。

## 当前架构

- 后端：Flask
- 数据库：PostgreSQL
- 数据源：Tushare Pro
- 研究层：Factor Lab、ML Lab
- 运行入口：`web/app.py`、`scripts/*.py`

## 当前主链路

1. `scripts/init_db.py`
2. `scripts/run_validation_data_sync.py`
3. `scripts/run_factor_validation.py`
4. `scripts/run_ml_lab.py`
5. `scripts/run_system_readiness.py`
6. `scripts/run_web.py`

## 当前口径

- 系统资金中性：不根据账户资金规模过滤股票，不输出强制仓位。
- `Settings.strategy.total_capital` 仅作可选展示或回测默认参数，不参与选股排序。
- 生产 baseline 不改。
- 研究层单独迭代。
- 因子不会因为一次失败直接删除。
- 旧版文档里写的 SQLite、老目录结构、老选股脚本，不再作为当前执行依据。
