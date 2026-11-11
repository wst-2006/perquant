# A-Share Quant System Project Map And Research Audit

## Current State

The project is now a capital-neutral A-share stock selection and trading-assistance system. It should not output buy amounts, target position ratios, automatic order instructions, or forced portfolio allocations.

Completed or usable v1 components:

- Data backbone: Tushare Pro plus PostgreSQL tables for stock universe, daily quotes, daily basic data, moneyflow, factor snapshots, ML scores, positions, reviews, and backtests.
- Web workbench: Flask + Jinja + JavaScript dashboard with data status, market, sector, selection, Factor Lab, ML Lab, positions, reviews, and basic backtest surfaces.
- Factor Lab: factor snapshots, forward returns, Rank IC style reports, TopN returns, monotonicity checks, and walk-forward entry points.
- ML Lab: training-frame export, dual stable/aggressive score concept, LightGBM/XGBoost/CatBoost model training, walk-forward research harness, and GPU offline package flow.
- First GPU full baseline: complete and archived locally at `reports/ml_lab/research_harness_v1_full`.

Latest GPU baseline verdict:

- Integrity: 23 folds, 138 trained runs, 18 features, full `20240102-20260601` frame.
- Stable target: weak signal, observe only.
- Aggressive target: weak Top10/Top20 signal, observe only.
- Production use: rejected for now; do not wire these models into live ranking.

## Highest-Uncertainty Questions

### 1. Are labels and simulated trade returns truly tradable?

This is the highest-risk question. If `label_net_return`, `label_stable_positive`, or `label_top10` include future information, unrealistic entry/exit prices, or data that was unavailable at signal time, then model quality is meaningless.

Audit requirements:

- Signal date, feature date, buy date, and sell date must be explicit for every row.
- Features must be computed only from data available at signal time.
- A-share constraints must be represented: T+1, limit-up entry blocks, limit-down exit blocks, suspension/no-price cases, board lot size, fees, stamp tax, and slippage.
- Financial-statement features must later use announcement dates, not report-period end dates, before becoming production features.
- Cross-sectional normalization must not use validation or future rows beyond the signal date.

Current status:

- The ML backtest simulates next-day open entry, one-lot trades, limit-up entry blocking, exit quote search, fees, stamp tax, and slippage.
- The research harness uses walk-forward time splits, which is directionally right.
- A dedicated training-frame label audit now exists at `reports/ml_lab/research_audit_20260708/research_audit_report.json`.
- The first label audit passed basic checks: no missing required columns, no date-order violations, no suspicious feature names, and no sensitive columns.
- PostgreSQL raw-quote reconstruction audit now exists at `reports/ml_lab/label_reconstruction_audit_20260709/label_reconstruction_audit.json`.
- Reconstruction audit sampled 2,000 rows and passed: missing quote rows `0`, tradable mismatches `0`, return mismatches `0`, mismatch rate `0.0`.
- Non-tradable reasons found in the sample were realistic: limit-up entry blocks `39`, limit-down exit blocks `15`.
- Moneyflow feature missingness remains material at about `26.66%`.

### 2. Does the model beat simple baselines across market regimes?

The first GPU result shows weak positive ranking signal but does not yet prove economic usefulness.

Audit requirements:

- Compare against random TopN from the same tradable universe.
- Compare against existing rule scores (`stable_score`, `aggressive_score`) as baselines.
- Compare against simple single-factor baselines, such as return_5d, return_20d, volume_ratio, and industry-relative strength.
- Report train-vs-validation performance to expose overfit gaps.
- Split results by market regime and identify explicit disable conditions.

Current status:

- Walk-forward validation exists.
- Top10/20/50 consistency checks exist and correctly flag weak broader TopN behavior.
- Random, rule, and single-factor baselines now exist through `python scripts/run_ml_lab.py audit-research`.
- First baseline audit: best random Top10 average return was `-0.001138`; all rule and simple-factor baselines were rejected.
- Evaluation-regime audit now exists in `reports/ml_lab/research_audit_20260708/research_audit_report.json`.
- Current evaluation regimes: broad down `242` days, broad up `196` days, mixed `143` days. This is an outcome-based research diagnostic, not a live trading signal.
- Baseline regime summary and failure windows are exported to `reports/ml_lab/research_audit_20260708/baseline_regime_summary.csv` and `reports/ml_lab/research_audit_20260708/baseline_failure_windows.csv`.
- The first GPU models beat simple baselines directionally but remain `observe_only`.
- Train-vs-validation overfit diagnostics have been added for future harness runs; the already-completed GPU run does not contain train metrics.

## Biggest Missing Layer

The biggest missing layer is a research audit and decision-gate system.

This layer should sit between research outputs and the Web workbench. Its job is not to produce more scores; its job is to decide whether a score is trustworthy enough to show as a candidate signal.

Minimum gate states:

- `rejected`: do not display as model signal.
- `observe_only`: may display in research pages, not production candidate ranking.
- `candidate`: may enter paper/live observation with strict tracking.
- `usable`: may affect candidate sorting, still never generates buy amount or target position.

Minimum gate checks:

- Data coverage and missingness.
- Label/tradability audit.
- Leakage audit.
- Random/rule/simple-factor baseline comparison.
- Walk-forward fold stability.
- Market-regime stability.
- Drawdown and failure-window review.
- Versioned report and artifact manifest.

## Project Map

### 1. Data Layer

Purpose: collect real A-share data without fabrication.

Status: v1 usable.

Main gaps:

- Longer historical sync for `2022-2026` or earlier.
- Better coverage reports by date, field, and universe.
- Financial statement announcement-date handling before using fundamentals in ML labels/features.

### 2. Data Audit Layer

Purpose: prove data is complete enough and honest enough for research.

Status: partial.

Needed next:

- Daily coverage audit for quotes, daily_basic, moneyflow, industry data.
- Missingness report by feature and date.
- Tradability coverage report for suspensions, limit-up/down, and no-price rows.

### 3. Label And Trading Assumption Layer

Purpose: define realistic future returns for model training.

Status: partial and high priority.

Needed next:

- Explicit label policy document.
- Failure-case counts for limit-up entry, limit-down exit, missing price, and non-tradable rows.
- Expand reconstruction audit from sampled validation to recurring report generation before every promoted training run.

### 4. Factor And Feature Layer

Purpose: create explainable predictive variables.

Status: v1 thin.

Needed next:

- Expand from 18 features to roughly 60-120 controlled features.
- Add multi-period momentum/reversal, cumulative moneyflow, industry ranks, market-regime interactions, volatility shifts, liquidity, and tradability features.
- Record feature-set version separately from model version.

### 5. Research And Model Layer

Purpose: test whether models add value over baselines.

Status: pipeline works, current model signal weak.

Needed next:

- Promote random/rule/single-factor baselines into every GPU report package.
- Use train-vs-validation metrics in the next GPU run to measure overfit gaps.
- Include regime-conditioned summaries and failure-window exports in every GPU report package.
- Feature importance and permutation diagnostics.

### 6. Observation Layer

Purpose: track model and rule candidates without automatic trading.

Status: concept exists; tracking is not complete.

Needed next:

- Daily TopN snapshot tracking.
- Subsequent 1/3/5/10/20 day outcome tracking.
- Manual decision log: selected, ignored, reason, actual result.
- Model disable-condition tracking.

### 7. Workbench Layer

Purpose: support manual trading review.

Status: v1 usable.

Needed next:

- Make research gate visible: rejected / observe_only / candidate / usable.
- Show model version, feature version, and latest audit verdict.
- Avoid presenting weak research output as production signal.

## External Method References

These references support the audit direction:

- scikit-learn `TimeSeriesSplit`: time-ordered validation prevents training on future observations. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Purging and embargoing in financial cross-validation: useful when labels overlap through time. https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/
- Common backtesting pitfalls: look-ahead bias, survivorship bias, transaction costs, and overfitting. https://portfoliooptimizationbook.com/book/8.2-seven-sins.html

## Immediate Execution Order

1. Fix low-risk system safety issues:
   - unknown `strategy_type` validation before Tushare/DB calls
   - Web model report source preference
   - legacy config warnings for target position and fixed buy amount files
2. Add regime-conditioned summaries and feature-importance diagnostics.
3. Create Feature Set v2 and export a new training frame.
4. Run local smoke, then GPU v2 with expanded features, baseline reports, label reconstruction audit, and train-vs-validation overfit diagnostics.
