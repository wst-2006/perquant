# 系统状态

当前仓库以 `src/quant_trading` 下的 Flask + PostgreSQL + Tushare 研究链路为准。系统定位为资金中性的选股与交易辅助工具：只输出候选股优先级、理由、风险和持仓观察，不替用户决定资金分配。

## 当前真实状态

- `src`：主链路可运行，包含数据同步、选股、Factor Lab、ML Lab、回测、持仓与复盘服务。
- `selection`：候选股排序不得使用账户资金、目标仓位或买入金额。
- `web`：Flask 入口可用，研究相关 API 已挂载。
- `config`：配置加载严格，缺少 `.env` 中的 `DB_PASSWORD` 或 `TUSHARE_TOKEN` 会阻断核心功能。
- `tests`：以当前实现为准，保留跑通性与纯逻辑测试。
- `docs`：存在历史文档与现状不一致的内容，执行时优先以本文件和代码为准。

## 执行口径

0. 资金分配、买入金额和加减仓幅度由用户人工决定。
1. 先保证系统能稳定跑通。
2. 再做因子验证和权重研究。
3. 不因单次失败删除因子，只记录版本和状态。

## 建议优先命令

```powershell
python scripts/init_db.py
python scripts/run_web.py
python scripts/run_factor_validation.py readiness
python scripts/run_ml_lab.py readiness
```
