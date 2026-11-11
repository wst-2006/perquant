# A股量化交易系统

个人 A 股量化选股与交易辅助系统。当前重建版以 Flask + PostgreSQL + Tushare Pro 为核心，第一阶段聚焦全市场真实数据选股、持仓观察、基准回测和 Web 工作台。系统资金中性：只筛选股票、解释理由和提示风险，不决定买入金额、仓位比例或加减仓幅度。

## 开发铁律

- 系统是选股与交易辅助工具，不是资金分配器。
- 候选股排序不得依赖账户总资金、资金池或目标仓位。
- 核心股票池必须通过 Tushare Pro 动态获取全市场 A 股。
- 核心资金流不得模拟、伪造或静态填充。
- 测试抽样只能放在 `tests/test_xxx.py` 或测试脚本中，不得改核心服务。
- DeepSeek/语义分析在当前阶段暂缓。

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

完整研究、GPU 与审计环境：

```powershell
pip install -r requirements-dev.txt
```

编辑 `.env`，填入 `TUSHARE_TOKEN` 和 PostgreSQL 连接信息。

初始化数据库：

```powershell
python scripts/init_db.py
```

启动 Web：

```powershell
python scripts/run_web.py
```

访问 `http://127.0.0.1:5000`。

盘后运行 v8 全市场观察链路：

```powershell
python scripts/run_v8.py daily-run --trade-date YYYYMMDD
```

该命令按顺序执行真实 Tushare 全市场同步、股票池审计、信号日快照门禁、生产地基门禁、`factor_lab_v2` 因子快照、旧信号结算和 v8 评分，并在 `reports/ml_lab/v8_daily` 保存带 SHA256 的版本化日报。数据已同步但上次在后续阶段中断时，可用 `--skip-sync` 从数据库断点恢复；所有门禁仍会执行。

构建独立的 D+2 短周期执行标签旁路数据集：

```powershell
python scripts/run_v8.py augment-short-labels --source-dir reports/ml_lab/training_frame_v9_events --output-dir reports/ml_lab/short_execution_labels_v1
```

该标签按 D+1 日开盘加滑点代理 09:35 买入，D+1 不允许卖出；D+2 起执行 `+1%/-5%`，同一日线双触发按止损优先，D+5 收盘终止且跌停不可卖时延期。分钟成交价和最低佣金没有审计数据，不会被伪造。

复核现有 v11/v10 在该执行口径下的隔离 OOS 表现：

```powershell
python scripts/run_v8.py audit-short-execution
```

运行独立短周期五头 walk-forward 研究：

```powershell
python scripts/run_v8.py short-execution-research --models lightgbm --device gpu
```

独立头只有通过最终隔离审核后才允许打包；审核失败时网站保持“D+2 独立预测头尚不可用”。

运行全市场选股：

```powershell
python scripts/run_selection.py --strategy short_term_core
python scripts/run_selection.py --strategy aggressive
python scripts/run_system_readiness.py
```

查看 v8 当前状态和某日诊断记录：

```powershell
python scripts/run_v8.py status
python scripts/run_v8.py candidates --signal-date YYYYMMDD
```

## 测试

```powershell
pytest
```

没有 `.env` 中的真实 Tushare token 时，真实数据集成测试会跳过；核心配置和纯计算测试仍会运行。

## 仓库结构

```text
src/quant_trading/   核心包：config / data / db / factor_lab / ml_lab / services / analysis / utils
scripts/             命令行入口：数据同步、因子快照、v8 日跑、回测、实验打包与审计（34 个脚本）
web/                 Flask Web 工作台：api_v1.py、templates、static
config/              配置模块，凭据一律从环境变量读取，不硬编码
migrations/          Alembic 数据库迁移
tests/               pytest 用例
docs/                设计文档、研究记录、操作指南
```

## 项目文档（docs/handoff/）

| 文件 | 内容 |
| --- | --- |
| `A股个人量化选股系统重建计划.md` | 系统重建目标、Phase 0 口径收口、模块与文档分层要求 |
| `A股实盘股票打分系统进度与后续总地图.md` / `.docx` | 当前进度（数据底座、Web、Factor Lab、ML Lab）与后续路线；docx 为原始排版 |

两份文档中的远程服务器地址、端口与登录用户名均已替换为 `****`。

## 配置与密钥

`config/` 与 `src/` 中没有任何硬编码凭据，全部通过 `.env` 注入。仓库里的 `.env.example` 只保留键名结构，**所有密钥取值一律以 `****` 占位**；`.env` 本身被 `.gitignore` 排除。

## 仓库不包含的内容

仓库刻意排除了体积大或与机器绑定的内容，克隆后按 `scripts/` 下的命令即可重新生成：

| 排除项 | 原因 |
| --- | --- |
| `reports/` | 本地约 32 GB 的回测、因子快照与训练产物（parquet / joblib） |
| `logs/`、`cache/`、`catboost_info/` | 运行日志与缓存 |
| `.venv/`、`__pycache__/`、`.pytest_cache/` | 本地环境与缓存 |
| `.env` | 真实凭据 |
| 根目录下的原始 `.docx` 与 `docs/_docx_extracts/` | 含远程训练服务器地址与登录方式，未直接入库；`docs/handoff/` 提供脱敏正文 |
