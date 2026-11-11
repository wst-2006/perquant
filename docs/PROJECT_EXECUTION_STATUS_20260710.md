# A 股实盘股票打分系统执行状态 2026-07-10

## 当前结论

系统数据地基已经补齐到可进入正式训练/上线评分的门槛。

训练门禁现在是 `pass`，原因是逐股财务慢接口 `fina_audit` 与 `fina_mainbz_P/D/I` 的股票覆盖率都已达到 90% 以上。这比“端点有几行数据就放行”的旧口径更严格，也更符合实盘研究要求。

已补充“双轨推进”门禁：

- `production_gate`: 仍严格要求完整数据地基，当前 `pass`，允许发布、打包正式模型和进入实盘评分版本。
- `diagnostic_gate`: 允许在当前特征真实依赖通过时运行诊断 baseline。所有输出必须标记 `diagnostic_only=true`、`promotion_allowed=false`。
- 当前 `factor_lab_v2` 的特征依赖为行情、`daily_basic`、`moneyflow`、指数、涨跌停、复权与 `fina_indicator`。`fina_audit/mainbz` 保留在生产地基要求中，但不是当前 v2 特征直接依赖。

## 已完成

- 增加股票池污染治理：
  - 非 A 股资产占位识别：ETF、指数、B 股、上交所产品/规则代码、北交所非 A 股形态。
  - `quarantine-non-stock-placeholders` 命令，默认 dry-run，`--apply` 后仅打隔离标签，不删除数据。
  - `repair-unknown-a-share-metadata` 命令，用 Tushare `namechange/bak_basic` 修复 UNKNOWN A 股元数据，不凭空造名称。
- 已执行股票池修复：
  - 隔离 `2154` 个非 A 股 UNKNOWN 占位为 `list_status=N`。
  - 修复 `300114.SZ` 为 `中航成飞`，来源为 Tushare 辅助端点。
  - 当前 `stock-universe-audit` 为 pass，未解决 UNKNOWN 为 `0`。
- 增强训练门禁：
  - `foundation_gate` 现在同时检查股票池审计。
  - `foundation_status` 增加 raw 财务端点按股票覆盖率检查。
  - 核心财务 raw 覆盖率阈值为 `90%`。
  - 增加特征感知 `diagnostic_gate`，诊断产物不可晋升。
  - `research-harness` 训练折增加标签 embargo：训练样本的 `label_buy_date/label_sell_date` 触碰验证窗口时会被剔除，避免时间切分污染。
  - 新增 `diagnostic-preflight` 命令，可不训练直接检查 foundation gate、训练帧股票池、label audit 与 walk-forward split audit。
- 增强财务同步：
  - `financial-foundation` 默认跳过已有 raw 记录。
  - 续跑时已跳过记录不再消耗 `--limit`。
  - 批次开始时预加载已有 raw 覆盖集合，避免每个请求查一次数据库。
- 已继续同步真实 Tushare 数据：
  - `fina_audit` 覆盖约 `5500 / 5607` 只样本股票。
  - `fina_mainbz_P/D/I` 覆盖约 `5644 / 5553 / 5098` 只样本股票。

## 当前数据状态

日期窗口：`20240401 - 20260601`

- `daily_quotes`: 100%
- `daily_basic`: 100%
- `moneyflow`: 100%
- `index_daily`: 108.22%（包含额外指数日期/代码）
- `stock_limit`: 100%
- `adj_factor`: 100%

raw 财务股票覆盖：

- `income`: 98.54%
- `balancesheet`: 98.54%
- `cashflow`: 98.54%
- `fina_indicator`: 98.48%
- `dividend`: 98.04%
- `fina_audit`: 98.09%
- `fina_mainbz_P`: 100.66%
- `fina_mainbz_D`: 99.04%
- `fina_mainbz_I`: 90.92%

## 诊断 baseline 状态

诊断训练帧：

- 文件：`reports/ml_lab/training_frame_v2_diagnostic_smoke.csv`
- manifest：`reports/ml_lab/training_frame_v2_diagnostic_smoke.manifest.json`
- 行数：`84307`
- 唯一股票：`5272`
- 日期：`20240401 - 20240424`
- 模式：`run_mode=diagnostic`
- 晋升：`diagnostic_only=true`，`promotion_allowed=false`

诊断预检：

- 报告：`reports/ml_lab/diagnostic_preflight_v2_smoke.json`
- universe audit：`pass`
- label audit：`pass`
- split audit：`pass`
- fold 0 标签 embargo：`8699` 行，embargo 后重叠 `0`
- fold 1 标签 embargo：`10338` 行，embargo 后重叠 `0`

label reconstruction audit：

- 报告：`reports/ml_lab/label_reconstruction_audit_v2_diagnostic_smoke/label_reconstruction_audit.json`
- 状态：`pass`
- 抽样：`300`
- tradable mismatches：`0`
- return mismatches：`0`

诊断 research harness：

- 报告：`reports/ml_lab/research_harness_v2_diagnostic_smoke/research_harness_report.json`
- fold：`2`
- feature_count：`113`
- 模型：`lightgbm`
- 训练折已启用标签 embargo
- 结果仍是诊断烟测，不能代表生产可用性。

诊断 baseline audit：

- 报告：`reports/ml_lab/research_audit_v2_diagnostic_smoke/research_audit_report.json`
- gate：`observe_only`
- label audit：`pass`
- split audit：`pass`
- 最强简单 baseline：`baseline:return_20d` top10，`avg_mean_net_return=0.020206`
- 当前模型最好摘要之一：`lightgbm aggressive top20`，`avg_mean_net_return=0.012613`
- 结论：在这个极小烟测窗口内，模型没有稳定跑赢简单动量 baseline。不得上线，不得包装为正式评分模型。

## 当前状态

`foundation-status` 现在应显示：

- `status`: `pass`
- `blocking`: `[]`
- `low_raw_stock_coverage`: `[]`

可以进入正式训练/上线评分链路，但诊断产物仍然必须保留 `diagnostic_only` 标记。

## 继续执行命令

分批继续跑，建议每次 500 有效请求：

```powershell
python scripts\sync_data.py financial-foundation --start-date 20240401 --end-date 20260601 --endpoints fina_audit --mode stock --limit 500
python scripts\sync_data.py financial-foundation --start-date 20240401 --end-date 20260601 --endpoints fina_mainbz --mode stock --limit 500
python scripts\sync_data.py foundation-status --start-date 20240401 --end-date 20260601
```

接口已验证可用，当前失败数为 0。

最近一轮慢端点补齐结果：

- `fina_audit`: 多轮补齐后覆盖到 `5500 / 5607`
- `fina_mainbz`: 多轮补齐后覆盖到 `P/D/I = 5644 / 5553 / 5098`
- 补齐后 `foundation-status` 已转为 `pass`

诊断 baseline 前置检查命令：

```powershell
python scripts\run_ml_lab.py diagnostic-preflight --training-frame-path reports\ml_lab\training_frame_v2_diagnostic_smoke.csv --output-path reports\ml_lab\diagnostic_preflight_v2_smoke.json --train-days 8 --validation-days 4 --step-days 4 --top-n 10,20 --gate-mode diagnostic --gate-factor-set factor_lab_v2
```

## 验证

本轮最终验证：

```text
python -m compileall -q src web scripts tests
python -m pytest tests\test_ml_research_harness.py tests\test_foundation_gate.py tests\test_ml_gpu_experiment_artifacts.py tests\test_data_sync_helpers.py -q
29 passed
python -m pytest -q
99 passed
```

## 最终结论

- 生产数据地基已达标。
- 诊断门禁仍有效，且诊断产物明确不可晋升。
- 下一步可以转正式训练和更大范围的研究审查。
