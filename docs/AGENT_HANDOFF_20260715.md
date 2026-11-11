# A股 Top5 机会检索系统：交接说明（2026-07-15）

## 0. 交接结论

工程链路可运行，研究结论尚未达到可投入真实资金的标准。系统当前正确状态是 `diagnostic_only` / `observe_only`：可以每天冻结候选并开始长期纸面观察，不能包装成可用策略，更不能自动交易。

本轮已完成 v11 双头 Top5 GPU 研究、v10 市场状态准入回放，以及 v12 纸面账本的数据结构。不要将模型选择期的高收益误作最终 OOS 表现；最终隔离审核不支持晋升。

## 1. 不可破坏的硬约束

- 全市场动态 A 股股票池，来自真实 Tushare Pro；不得人为限定核心股票池。
- 不得伪造/模拟行情、资金流、事件或公告数据；缺失即缺失。
- 信号固定为 D 日收盘后计算，D+1 开盘前冻结研究候选；只允许人工最终筛选。
- 所有特征、事件和市场状态只能使用信号时点已经公开的数据。
- 回测必须维持 A 股 T+1、停牌、真实涨跌停、交易可达性和费用约束。
- 候选检索能力、默认纸面组合和人工选择必须分账，不互相污染标签或指标。
- 不输出自动下单、仓位、金额或交易指令。70/30 仅是纸面组合的百分比结构。
- 测试/简化逻辑只能放在 `tests/test_xxx.py` 或独立测试工具，不得篡改核心研究链路。

## 2. 当前可验证状态

2026-07-15 已执行：

```powershell
python -m compileall -q src web scripts tests
python -m pytest -q
```

结果：编译通过，`172 passed`。

项目不是 Git 仓库。交接应复制整个项目目录，尤其保留 `reports/`、`config/`、`src/`、`scripts/`、`tests/`、`web/`、`.env`（仅本机私密保管，绝不外发 token）与本文件。

## 3. 已完成模块

| 模块 | 结论 | 主要位置 |
| --- | --- | --- |
| 数据地基/全市场审计 | 已具备并曾通过生产门禁 | `scripts/run_v8.py`、`src/quant_trading/ml_lab/v8_audit.py` |
| v8/v9 标签、事件、138 维特征 | 已具备真实端点、T+1 与标签重建审计 | `v8_labels.py`、`v8_events.py`、`v9_events.py` |
| GPU walk-forward 排序研究 | 已完成 LightGBM/XGBoost 全量研究与基线/失败窗口报告 | `v8_research.py` |
| v11 稳定/进攻双头重排 | 已完成第一轮全量 GPU 研究 | `v8_research.py`、报告见下文 |
| v10 市场状态准入 | 已完成对 v11 OOS 日度流的隔离回放 | `v10_market_gate.py` |
| v12 纸面观察数据结构 | 已接入研究 Top5 / 默认 70:30 / 人工选择三本账 | `v8_service.py`、`v8_daily.py` |
| Web/API 基础 | 可运行；研究接口已有 | `web/`、`v8_service.py` |

## 4. 当前研究事实，禁止误读

### v11 双头全量 GPU 排序

报告：`reports/ml_lab/v11_dual_research_gpu_lgb_xgb_full/v8_research_report.json`

- 最优候选：`ensemble`，120 日训练窗，`v11_dual_70_30`。
- 模型选择分区：`MeanNetReturn@5 = 2.0126%`，正收益折率 `88.89%`。这不是最终结论。
- 最终隔离审核：`MeanNetReturn@5 = 0.0809%`，正收益折率 `40%`，最大回撤 `-30.6429%`。
- 最终状态：`observe_only`。
- 未通过项：正收益折率、两个机会 Hit 指标相对基线、全亏损率改善、回撤改善、跨市场状态保护。

主要失败集中在 `2026-01` 到 `2026-03`，尤其 `20260227-20260326`。这是下一轮的核心研究对象，而非参数磨皮目标。

### v10 对 v11 OOS 的市场准入

报告：`reports/ml_lab/v10_market_gate_v11_dual_oos/v10_market_gate_report.json`

- 覆盖率 `17.5%`（7 / 40 日），低于 `25%` 门槛。
- 准入日平均净收益 `2.0241%`，全日历收益（`no_signal = 0`）`0.3542%`，准入日正收益率 `57.14%`。
- 日历最大回撤 `-5.7709%`，风险压制有诊断价值。
- 最终状态仍为 `observe_only`：覆盖率、模型选择门禁、相对无准入和多类基线的胜出均未完成。

因此 v10 显示“少参与可能有帮助”，但样本过少，不能把它当可用市场择时模型。

## 5. 最近代码变更（必须保留）

- `v8_research.package_market_gate` 能打包 v10 研究报告。
- `v8_service` 可评估 `eligible_stable`、`eligible_aggressive`、`no_signal`，诊断模型输出带 `diagnostic_` 前缀，不能进入默认纸面组合。
- 每日评分结果有 `paper_ledger`，包含 `research_top5`、`default_70_30_portfolio`、`manual_selection`。
- 候选含 `candidate_type`、稳定/进攻分数和证据字段。
- 已补充 v10/v12 行为测试；当前总测试数为 172。

## 6. 建议的下一步顺序

不要继续围绕现有 v11 反复调参数。新 agent 应按以下顺序推进：

1. 使用当前 v11 + v10 研究报告训练并打包一个明确标记为诊断的工件，验证每日链路从工件加载到冻结日报的完整性。
2. 运行每日冻结候选与三本账纸面观察；所有结果仍为 `diagnostic_only`，不可产生真实资金建议。
3. 为 v10 失败窗口做错误类型分析和严格消融：状态特征、市场广度、涨跌停压力、行业扩散度、成交额变化。只允许训练期决定阈值。
4. 再设计一轮 v10 状态准入，使覆盖率、基线比较和最终隔离审核都能经受检验；若不能，保留 no-signal 规则作为诊断，不晋升。
5. 仅在 v10 隔离 OOS 成为 `candidate` 后，进入 v12 的 40 至 100 个有效信号日纸面验证；同时继续维护不可改写的快照与结算。
6. 在任何候选晋升之前，冻结一个从未参与设计的最终盲测区间。失败不能回写该盲测期调参。

## 7. 可复跑命令

完整代码验证：

```powershell
python -m compileall -q src web scripts tests
python -m pytest -q
```

打包 v11 诊断工件（状态应保持 `diagnostic_only`）：

```powershell
python scripts/run_v8.py train-artifact `
  --dataset-dir reports/ml_lab/training_frame_v9_events `
  --research-report reports/ml_lab/v11_dual_research_gpu_lgb_xgb_full/v8_research_report.json `
  --output-dir reports/ml_lab/v11_dual_artifact `
  --market-gate-report reports/ml_lab/v10_market_gate_v11_dual_oos/v10_market_gate_report.json `
  --models lightgbm,xgboost `
  --device gpu `
  --n-estimators 120 `
  --dataset-audit-path reports/ml_lab/v9_events_dataset_audit/v8_dataset_audit.json
```

对指定日做已同步数据的诊断日报：

```powershell
python scripts/run_v8.py daily-run --trade-date YYYYMMDD --artifact-path reports/ml_lab/v11_dual_artifact/artifact.json --skip-sync
```

若数据还未同步，移除 `--skip-sync`，让系统先从真实 Tushare 更新并跑所有门禁。

## 8. 交接时需要用户提供的内容

- 保留可用的 `TUSHARE_TOKEN` 与 PostgreSQL 配置（私密，不粘贴给外部服务）。
- 真实成交交割单/费用明细可在以后用来做前向费用校准；当前默认费率是保守研究假设，尚不构成阻塞。
- 集合竞价暂无合规、可审计数据源，因此保持人工执行信息，不得回填或伪造历史竞价过滤。

## 9. 当前绝对禁止的结论

- 不得说策略已可实盘盈利或已可正式投入资金。
- 不得因模型选择期指标好看而忽略最终隔离审核。
- 不得因人工二筛存在而降低系统 Top5 的独立质量标准。
- 不得为提高覆盖率而放宽时间隔离、可交易性或股票池审计。
