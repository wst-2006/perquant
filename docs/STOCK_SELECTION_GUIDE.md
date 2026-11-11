# 选股研究指南

当前仓库的选股研究分两层：

1. 传统规则选股
2. Factor Lab + ML Lab 双评分研究

## 当前真实状态

- 传统规则选股服务仍在。
- 研究主线已转为 `Factor Lab -> ML Lab -> Top10 -> Backtest`。
- 选股主流程资金中性，不因账户资金规模、一手金额或目标仓位改变排序。
- 选股结果不应一次性定稿，必须保留实验记录。

## 当前执行命令

```powershell
python scripts/run_selection.py --strategy short_term_core
python scripts/run_selection.py --strategy aggressive
python scripts/run_ml_lab.py top10 --trade-date 20260601
python scripts/run_ml_lab.py backtest --start-date 20250601 --end-date 20260520
```

## 研究规则

- 不模拟资金流。
- 不输出建议买入金额或目标仓位作为系统结论。
- 不用一次验证直接定最终因子。
- 不删除失败结果，只改状态和版本。
