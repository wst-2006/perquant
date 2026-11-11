# 仓库真实状态审计表

审计依据：当前代码、脚本入口、测试文件、报告产物与已读文档。

## 总览

| 分组 | 状态 | 结论 |
|---|---|---|
| `src` | 部分可用 | 主链路存在并能运行，但研究结果尚未收敛，旧测试和旧文档曾与实现脱节。 |
| `web` | 可运行 | Flask 应用和核心 API 已挂载，个别页面文案曾与当前研究版本不一致，已开始收口。 |
| `config` | 部分可用 | `.env`、Tushare、PostgreSQL 是硬依赖，生产参数和研究参数仍需继续分层。 |
| `tests` | 部分可用 | 纯逻辑和接口冒烟测试可用，完整真实数据链路测试仍不足。 |
| `docs` | 冲突收口中 | 多份历史完成度文档过时，当前执行以 `SYSTEM_STATUS.md`、`QUICK_START.md` 和本文件为准。 |

## 具体问题

### `src`
- 数据同步、Factor Lab、ML Lab、回测、Web API 已具备闭环入口。
- `SelectionService` 仍是逐股实时拉取 Tushare 的生产选股路径，速度和频率压力较大。
- ML Lab 当前双评分与回测是研究层，不代表最终策略可用。
- 资金流缺失会被惩罚，不应模拟或填假值。

### `web`
- `/api/health` 可作为服务可用性检查。
- `/api/factor/readiness`、`/api/ml/readiness` 可作为研究链路准备度检查。
- `/api/selection/run`、`/api/backtest/buy_hold` 依赖真实 Tushare/数据库环境，不适合作为无数据单测。

### `config`
- `.env.example` 给出必要字段，但真实运行必须配置 `TUSHARE_TOKEN` 和 `DB_PASSWORD`。
- `src/quant_trading/config.py` 是当前权威配置入口。
- `config/` 下部分旧配置文件仅作历史参考，不再参与主运行口径。

### `tests`
- 当前优先保留跑通性测试：配置、Web health、Factor Lab 纯逻辑、ML Lab 纯逻辑。
- 缺少针对完整研究链路的可重复测试夹具。
- 后续应补：固定小样本数据库、Top10 重复运行、回测边界条件。

### `docs`
- `SYSTEM_STATUS.md` 已收口为当前真实状态。
- `QUICK_START.md` 是当前启动参考。
- `SYSTEM_KNOWLEDGE.md`、`WORKFLOW_GUIDE.md`、`STOCK_SELECTION_GUIDE.md` 含旧系统路径，执行时不能直接照搬。

## 下一步

1. 跑测试，修复当前实现和测试继续暴露出的断点。
2. 保留已有报告产物，建立版本化实验记录。
3. 再进入因子候选池扩展、验证和权重收敛。
