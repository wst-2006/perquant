# 工作流

当前工作流以研究链路为主，不再沿用旧版 `select_stocks_clean.py` / `daily_exit_check.py` 叙述。`config/` 下的旧策略文件也不再作为主执行口径。

## 推荐顺序

1. 配置 `.env`
2. 初始化数据库
3. 跑数据同步
4. 跑 Factor Lab readiness / evaluate
5. 跑 ML Lab readiness / top10 / backtest
6. 用 Web 查看结果

## 当前推荐命令

```powershell
python scripts/init_db.py
python scripts/run_validation_data_sync.py
python scripts/run_factor_validation.py readiness
python scripts/run_ml_lab.py readiness
python scripts/run_system_readiness.py
python scripts/run_web.py
```

## 原则

- 系统只负责筛选股票、解释理由、提示风险和辅助复盘。
- 资金分配、买入金额、加减仓比例由用户人工决定。
- 先跑通，再谈因子收敛。
- 所有实验结果保留版本。
- 以当前代码和测试为准，不以旧完成度文档为准。
