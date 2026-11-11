# ML Research Harness 20260708 Review

## Run Integrity

- Source package: `reports/ml_lab/ml_research_result_20260708_212119.tar.gz`
- SHA256: `8F388816C5D9E8273E0A9E0375E44B0FC4A4E994304C6EE22B4BEB9B768E40B9`
- GPU log dir: `logs/gpu_research_full_20260708_212119`
- Output dir: `reports/ml_lab/research_harness_v1_full`
- Exit code: `0`
- Training frame: `reports/ml_lab/training_frame_full.csv`
- Config: train 120 days, validate 20 days, step 20 days, TopN 10/20/50
- Models: LightGBM, XGBoost, CatBoost
- Targets: stable, aggressive
- Fold count: 23
- Result count: 138
- Status: all 138 runs trained
- Feature count: 18

## High-Level Verdict

This first full baseline is valid as a research run, but it is not strong enough for production scoring.

The models show weak positive ranking signal in selected Top10/Top20 slices, especially in favorable windows, but the signal is not broad or stable enough. Top20/Top50 frequently weaken, drawdowns are large for short-cycle use, and average win rates remain below 50%.

Recommended status:

- `stable`: observe, not usable
- `aggressive`: observe, not usable
- production integration: do not promote to live scoring
- next step: feature engineering v2 and stricter market-regime conditioning

## Summary Highlights

Best-looking aggressive slices:

- `lightgbm / aggressive / top10`: positive fold rate `0.608696`, average net return `0.000553`, average win rate `0.478696`, average max drawdown `-0.136727`
- `catboost / aggressive / top20`: positive fold rate `0.608696`, average net return `0.000272`, average win rate `0.465435`, average max drawdown `-0.126275`
- `catboost / aggressive / top10`: positive fold rate `0.565217`, average net return `0.000229`, average win rate `0.465217`, average max drawdown `-0.141645`

Best-looking stable slices:

- `xgboost / stable / top10`: positive fold rate `0.521739`, average net return `0.000277`, average win rate `0.473478`, average max drawdown `-0.120329`
- `lightgbm / stable / top10`: positive fold rate `0.521739`, average net return `0.000080`, average win rate `0.461304`, average max drawdown `-0.111946`
- `catboost / stable / top10`: positive fold rate `0.478261`, average net return `0.000423`, average win rate `0.471087`, average max drawdown `-0.132350`

## Key Risk Findings

- No slice reached the current `usable` threshold.
- Several Top10 slices are positive while Top20/Top50 are weak; the harness correctly tags this as `top10_positive_but_broader_topn_weak`.
- Average win rates are around `46%-48%`, not enough for a confident short-cycle trading assistant.
- Worst fold drawdowns are large. For example:
  - `lightgbm / aggressive / top10`, fold 5: average net return `-0.030909`, max drawdown `-0.470894`
  - `catboost / aggressive / top20`, fold 5: average net return `-0.036126`, max drawdown `-0.525751`
  - `catboost / stable / top10`, fold 5: average net return `-0.014460`, max drawdown `-0.268098`
- Fold 5 (`2024-12-02` to `2024-12-27`) is a major failure window across models.
- Fold 19-20 (`2026-01-26` to `2026-03-30`) also weakens several aggressive slices.

## Interpretation

The current 18-feature baseline has some market-dependent predictive signal, but the signal is too shallow and unstable. It should be treated as proof that the pipeline works, not proof that the current factor set is ready for live model scoring.

The next research iteration should focus on:

- stronger multi-period momentum and reversal features
- 3/5/10 day cumulative moneyflow features
- industry-relative ranks, not only raw relative strength
- market-regime gating and separate model evaluation by regime
- liquidity, limit-up/down, and tradability features
- feature importance and permutation diagnostics
- comparison against simple rule baselines and random TopN baselines

## Recommended Next Actions

1. Archive this run as `observe_only`.
2. Do not wire these models into production candidate ranking yet.
3. Fix small system issues before new experiments:
   - validate unknown `strategy_type` before Tushare/DB calls
   - make Web model report point to the latest versioned or harness report source
   - mark legacy position/decision config as historical reference
4. Build Feature Set v2 and export a new training frame.
5. Run a smaller local smoke test, then package a second GPU experiment.

## Follow-Up Research Audit

Audit command:

```powershell
python scripts/run_ml_lab.py audit-research --training-frame-path reports/ml_lab/training_frame_full.csv --output-dir reports/ml_lab/research_audit_20260708 --model-report-path reports/ml_lab/research_harness_v1_full/research_harness_report.json --train-days 120 --validation-days 20 --step-days 20 --top-n 10,20,50
```

Generated files:

- `reports/ml_lab/research_audit_20260708/research_audit_report.json`
- `reports/ml_lab/research_audit_20260708/baseline_summary.csv`
- `reports/ml_lab/research_audit_20260708/baseline_regime_summary.csv`
- `reports/ml_lab/research_audit_20260708/baseline_failure_windows.csv`
- `reports/ml_lab/label_reconstruction_audit_20260709/label_reconstruction_audit.json`
- `reports/ml_lab/label_reconstruction_audit_20260709/label_reconstruction_mismatches.csv`

Label audit:

- Rows: `3,115,705`
- Dates: `581`
- Stocks: `5,524`
- Tradable rate: `0.973425`
- Mean tradable net return: `-0.001683`
- Stable positive rate: `0.451246`
- Top10 label rate: `0.099744`
- Date-order violations: `0`
- Suspicious feature columns: none
- Sensitive columns: none
- Status: `pass`

Raw quote reconstruction audit:

- Sample rows: `2,000`
- Status: `pass`
- Missing quote rows: `0`
- Tradable mismatches: `0`
- Return mismatches: `0`
- Mismatch rate: `0.0`
- Sample non-tradable reasons: limit-up entry blocks `39`, limit-down exit blocks `15`
- Interpretation: sampled training labels can be reconstructed from PostgreSQL daily quotes, so the previous failed audit was a date-format parsing issue rather than evidence of broken labels.

Feature coverage risks:

- `feature_main_net_to_amount` missing rate: `0.266639`
- `feature_main_net_to_circ_mv` missing rate: `0.266639`
- `feature_relative_industry_strength_20d` missing rate: `0.035416`
- `feature_return_20d` missing rate: `0.035416`
- `feature_return_5d` missing rate: `0.008863`

Baseline comparison:

- Best baseline is `random / top10`: positive fold rate `0.434783`, average net return `-0.001138`, average win rate `0.455`.
- Rule and simple-factor baselines are all rejected in this audit.
- The GPU models beat these simple baselines directionally, but still remain too weak for production.

Evaluation-regime diagnostics:

- Broad down days: `242`, average market mean return `-0.021065`, average positive rate `0.223572`
- Broad up days: `196`, average market mean return `0.020338`, average positive rate `0.732296`
- Mixed days: `143`, average market mean return `0.000421`, average positive rate `0.450867`
- This regime label is outcome-based and research-only. It must not be used as a live input signal.
- Baseline failure windows exported: `50`

Local validation boundary:

- Local validation covers code tests, syntax compilation, report generation, label reconstruction, and audit logic.
- Heavy model training remains a remote GPU task. A local pass does not prove remote GPU dependency compatibility or final model quality.
- Current local validation after the audit additions: `85 passed`; `python -m compileall -q src web scripts tests` passed.

Updated gate:

- Status: `observe_only`
- Meaning: research signal exists but is not strong enough for production ranking.
- Required next gate: Feature Set v2 plus recurring label reconstruction audit, overfit diagnostics, and regime-specific validation.
