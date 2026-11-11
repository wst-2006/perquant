from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import uuid

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Index,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


class Stock(Base):
    __tablename__ = "stocks"

    ts_code: Mapped[str] = mapped_column(String(16), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(12), index=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    area: Mapped[str | None] = mapped_column(String(64), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    market: Mapped[str | None] = mapped_column(String(32), nullable=True)
    list_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    list_status: Mapped[str] = mapped_column(String(1), default="L", index=True)
    is_hs: Mapped[str | None] = mapped_column(String(8), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DailyQuote(Base):
    __tablename__ = "daily_quotes"
    __table_args__ = (UniqueConstraint("ts_code", "trade_date", name="uq_daily_quote_stock_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    pre_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    change: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_chg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DailyBasic(Base):
    __tablename__ = "daily_basic"
    __table_args__ = (UniqueConstraint("ts_code", "trade_date", name="uq_daily_basic_stock_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover_rate_f: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    pe: Mapped[float | None] = mapped_column(Float, nullable=True)
    pe_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)
    pb: Mapped[float | None] = mapped_column(Float, nullable=True)
    ps: Mapped[float | None] = mapped_column(Float, nullable=True)
    ps_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)
    dv_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    dv_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    float_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    free_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_mv: Mapped[float | None] = mapped_column(Float, nullable=True)
    circ_mv: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MoneyflowDaily(Base):
    __tablename__ = "moneyflow_daily"
    __table_args__ = (UniqueConstraint("ts_code", "trade_date", name="uq_moneyflow_stock_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    buy_sm_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    sell_sm_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    buy_md_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    sell_md_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    buy_lg_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    sell_lg_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    buy_elg_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    sell_elg_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_mf_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class IndexDaily(Base):
    __tablename__ = "index_daily"
    __table_args__ = (UniqueConstraint("ts_code", "trade_date", name="uq_index_daily_code_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    pre_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    change: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_chg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StockAdjFactor(Base):
    __tablename__ = "stock_adj_factors"
    __table_args__ = (UniqueConstraint("ts_code", "trade_date", name="uq_adj_factor_stock_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    adj_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StockLimitDaily(Base):
    __tablename__ = "stock_limit_daily"
    __table_args__ = (UniqueConstraint("ts_code", "trade_date", name="uq_stock_limit_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    up_limit: Mapped[float | None] = mapped_column(Float, nullable=True)
    down_limit: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TushareRawRecord(Base):
    __tablename__ = "tushare_raw_records"
    __table_args__ = (UniqueConstraint("endpoint", "row_key", name="uq_tushare_raw_endpoint_row"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    endpoint: Mapped[str] = mapped_column(String(64), index=True)
    row_key: Mapped[str] = mapped_column(String(256), index=True)
    ts_code: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    trade_date: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    ann_date: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    f_ann_date: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    end_date: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    period: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FactorSnapshot(Base):
    __tablename__ = "factor_snapshots"
    __table_args__ = (UniqueConstraint("trade_date", "ts_code", "factor_set", name="uq_factor_snapshot_date_stock_set"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    factor_set: Mapped[str] = mapped_column(String(32), default="factor_lab_v1", index=True)
    industry: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    market_regime: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    market_cap_bucket: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    raw_factors: Mapped[dict] = mapped_column(JSON)
    normalized_factors: Mapped[dict] = mapped_column(JSON)
    data_quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FactorForwardReturn(Base):
    __tablename__ = "factor_forward_returns"
    __table_args__ = (UniqueConstraint("trade_date", "ts_code", "horizon", name="uq_forward_return_date_stock_horizon"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    horizon: Mapped[int] = mapped_column(Integer, index=True)
    forward_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_forward_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    exit_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    tradable: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    constraints: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FactorEvaluationRun(Base):
    __tablename__ = "factor_evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    factor_set: Mapped[str] = mapped_column(String(32), index=True)
    factor_name: Mapped[str] = mapped_column(String(64), index=True)
    start_date: Mapped[str] = mapped_column(String(8), index=True)
    end_date: Mapped[str] = mapped_column(String(8), index=True)
    horizon: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    summary: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FactorWalkForwardRun(Base):
    __tablename__ = "factor_walk_forward_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    factor_set: Mapped[str] = mapped_column(String(32), index=True)
    strategy_name: Mapped[str] = mapped_column(String(64), index=True)
    start_date: Mapped[str] = mapped_column(String(8), index=True)
    end_date: Mapped[str] = mapped_column(String(8), index=True)
    train_window: Mapped[int] = mapped_column(Integer, default=0)
    validation_window: Mapped[int] = mapped_column(Integer, default=0)
    step_window: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FactorWalkForwardFold(Base):
    __tablename__ = "factor_walk_forward_folds"
    __table_args__ = (UniqueConstraint("run_id", "fold_index", name="uq_factor_walk_forward_fold"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("factor_walk_forward_runs.id"), index=True)
    fold_index: Mapped[int] = mapped_column(Integer, index=True)
    train_start_date: Mapped[str] = mapped_column(String(8), index=True)
    train_end_date: Mapped[str] = mapped_column(String(8), index=True)
    validation_start_date: Mapped[str] = mapped_column(String(8), index=True)
    validation_end_date: Mapped[str] = mapped_column(String(8), index=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FactorCandidatePoolRun(Base):
    __tablename__ = "factor_candidate_pool_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    factor_set: Mapped[str] = mapped_column(String(32), index=True)
    pool_version: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(64), default="manual")
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FactorCandidatePoolItem(Base):
    __tablename__ = "factor_candidate_pool_items"
    __table_args__ = (UniqueConstraint("pool_run_id", "factor_name", "horizon", name="uq_factor_candidate_pool_item"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pool_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("factor_candidate_pool_runs.id"), index=True)
    factor_name: Mapped[str] = mapped_column(String(64), index=True)
    horizon: Mapped[int] = mapped_column(Integer, index=True)
    verdict: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    usable_windows: Mapped[int] = mapped_column(Integer, default=0)
    observe_windows: Mapped[int] = mapped_column(Integer, default=0)
    failed_windows: Mapped[str | None] = mapped_column(Text, nullable=True)
    rank_ic_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_ic_ir: Mapped[float | None] = mapped_column(Float, nullable=True)
    top50_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_v1_candidate: Mapped[bool] = mapped_column(Boolean, default=False)
    needs_more_tracking: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DataSyncRun(Base):
    __tablename__ = "data_sync_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_type: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    trade_date: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    start_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(8), nullable=True)
    affected_rows: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FactorScore(Base):
    __tablename__ = "factor_scores"
    __table_args__ = (UniqueConstraint("ts_code", "trade_date", "strategy_type", name="uq_factor_stock_date_strategy"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    strategy_type: Mapped[str] = mapped_column(String(32), index=True)
    total_score: Mapped[float] = mapped_column(Float)
    technical_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    capital_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sector_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_fit: Mapped[float | None] = mapped_column(Float, nullable=True)
    valuation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    growth_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SelectionRun(Base):
    __tablename__ = "selection_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    strategy_type: Mapped[str] = mapped_column(String(32), index=True)
    stock_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="created")
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    results: Mapped[list["SelectionResult"]] = relationship(back_populates="run")


class SelectionResult(Base):
    __tablename__ = "selection_results"
    __table_args__ = (UniqueConstraint("run_id", "rank", name="uq_selection_run_rank"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("selection_runs.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    score: Mapped[float] = mapped_column(Float)
    action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    run: Mapped[SelectionRun] = relationship(back_populates="results")


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(64))
    strategy_type: Mapped[str] = mapped_column(String(32), index=True)
    buy_date: Mapped[Date] = mapped_column(Date)
    buy_price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[int] = mapped_column(Integer)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    highest_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    lowest_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="holding", index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TradeLog(Base):
    __tablename__ = "trade_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(32))
    trade_date: Mapped[Date] = mapped_column(Date)
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReviewJournal(Base):
    __tablename__ = "review_journals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[Date] = mapped_column(Date, index=True)
    ts_code: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str] = mapped_column(String(32), default="daily", index=True)
    title: Mapped[str] = mapped_column(String(128))
    content: Mapped[str] = mapped_column(Text)
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_type: Mapped[str] = mapped_column(String(32), index=True)
    start_date: Mapped[str] = mapped_column(String(8), index=True)
    end_date: Mapped[str] = mapped_column(String(8), index=True)
    initial_capital: Mapped[float] = mapped_column(Float)
    final_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MlDualScore(Base):
    __tablename__ = "ml_dual_scores"
    __table_args__ = (UniqueConstraint("trade_date", "ts_code", "model_version", name="uq_ml_dual_score_date_stock_model"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    stable_score: Mapped[float] = mapped_column(Float)
    aggressive_score: Mapped[float] = mapped_column(Float)
    stable_rank: Mapped[int] = mapped_column(Integer, index=True)
    aggressive_rank: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MlTopCandidate(Base):
    __tablename__ = "ml_top_candidates"
    __table_args__ = (UniqueConstraint("trade_date", "ts_code", "list_type", "model_version", name="uq_ml_top_candidate"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    list_type: Mapped[str] = mapped_column(String(16), index=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    rank: Mapped[int] = mapped_column(Integer, index=True)
    stable_score: Mapped[float] = mapped_column(Float)
    aggressive_score: Mapped[float] = mapped_column(Float)
    contribution_factors: Mapped[list | None] = mapped_column(JSON, nullable=True)
    risk_tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MlV8CandidateSignal(Base):
    __tablename__ = "ml_v8_candidate_signals"
    __table_args__ = (UniqueConstraint("signal_date", "ts_code", "model_version", name="uq_ml_v8_candidate_signal"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_date: Mapped[str] = mapped_column(String(8), index=True)
    ts_code: Mapped[str] = mapped_column(String(16), ForeignKey("stocks.ts_code"), index=True)
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    rank: Mapped[int] = mapped_column(Integer, index=True)
    decision: Mapped[str] = mapped_column(String(32), index=True)
    overall_score: Mapped[float] = mapped_column(Float)
    p_hit_5pct_10d: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_net_return_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_mfe_10d: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_mae_10d: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_disagreement: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    risk_tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MlV8CandidateOutcome(Base):
    __tablename__ = "ml_v8_candidate_outcomes"
    __table_args__ = (UniqueConstraint("signal_id", "horizon", name="uq_ml_v8_candidate_outcome"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(Integer, ForeignKey("ml_v8_candidate_signals.id"), index=True)
    horizon: Mapped[int] = mapped_column(Integer, index=True)
    settlement_date: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    fixed_tradable: Mapped[bool] = mapped_column(Boolean, default=False)
    net_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    mfe_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit_5pct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hit_8pct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hit_12pct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MlV8ManualReview(Base):
    __tablename__ = "ml_v8_manual_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(Integer, ForeignKey("ml_v8_candidate_signals.id"), index=True)
    selected: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MlBacktestRun(Base):
    __tablename__ = "ml_backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    start_date: Mapped[str] = mapped_column(String(8), index=True)
    end_date: Mapped[str] = mapped_column(String(8), index=True)
    portfolio_rule: Mapped[str] = mapped_column(String(64), index=True)
    periods: Mapped[int] = mapped_column(Integer, default=0)
    trades: Mapped[int] = mapped_column(Integer, default=0)
    total_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_return_on_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MlExperimentRun(Base):
    __tablename__ = "ml_experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    experiment_type: Mapped[str] = mapped_column(String(32), index=True)
    start_date: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    end_date: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MlExperimentArtifact(Base):
    __tablename__ = "ml_experiment_artifacts"
    __table_args__ = (UniqueConstraint("experiment_run_id", "artifact_type", name="uq_ml_experiment_artifact"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("ml_experiment_runs.id"), index=True)
    artifact_type: Mapped[str] = mapped_column(String(32), index=True)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def _uuid_string() -> str:
    return str(uuid.uuid4())


class WebJob(Base):
    __tablename__ = "web_jobs"
    __table_args__ = (
        Index(
            "uq_web_jobs_active_dedupe_key",
            "dedupe_key",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running', 'cancel_requested')"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_string)
    job_type: Mapped[str] = mapped_column(String(64), index=True)
    dedupe_key: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    trade_date: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=1)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ResearchRun(Base):
    __tablename__ = "research_runs"
    __table_args__ = (
        UniqueConstraint("trade_date", "pipeline", "version", name="uq_research_run_date_pipeline_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_string)
    job_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("web_jobs.id"), nullable=True, index=True)
    trade_date: Mapped[str] = mapped_column(String(8), index=True)
    pipeline: Mapped[str] = mapped_column(String(64), default="v11_top5_daily", index=True)
    execution_mode: Mapped[str] = mapped_column(String(32), default="skip_sync", index=True)
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    decision: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    diagnostic_only: Mapped[bool] = mapped_column(Boolean, default=True)
    promotion_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
    data_version: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    audit: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    market_gate: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    paper_ledger: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    report_artifact: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ResearchCandidate(Base):
    __tablename__ = "research_candidates"
    __table_args__ = (
        UniqueConstraint("research_run_id", "rank", name="uq_research_candidate_run_rank"),
        UniqueConstraint("research_run_id", "ts_code", name="uq_research_candidate_run_stock"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    research_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("research_runs.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    candidate_type: Mapped[str] = mapped_column(String(32), default="stable", index=True)
    decision: Mapped[str] = mapped_column(String(64), index=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    stable_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    aggressive_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_hit_5pct_10d: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_net_return_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_mfe_10d: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_mae_10d: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_disagreement: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    risk_tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ManualSelection(Base):
    __tablename__ = "manual_selections"
    __table_args__ = (UniqueConstraint("research_candidate_id", name="uq_manual_selection_candidate"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    research_candidate_id: Mapped[int] = mapped_column(Integer, ForeignKey("research_candidates.id"), index=True)
    selected: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CandidateOutcome(Base):
    __tablename__ = "candidate_outcomes"
    __table_args__ = (UniqueConstraint("research_candidate_id", "horizon", name="uq_candidate_outcome_horizon"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    research_candidate_id: Mapped[int] = mapped_column(Integer, ForeignKey("research_candidates.id"), index=True)
    horizon: Mapped[int] = mapped_column(Integer, index=True)
    settlement_date: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    fixed_tradable: Mapped[bool] = mapped_column(Boolean, default=False)
    net_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    mfe_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit_5pct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hit_8pct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hit_12pct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database.url, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
    return _session_factory


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine, checkfirst=True)
    ensure_schema_v2()


def ensure_schema_v2() -> None:
    statements = [
        "ALTER TABLE daily_basic ADD COLUMN IF NOT EXISTS turnover_rate_f DOUBLE PRECISION",
        "ALTER TABLE daily_basic ADD COLUMN IF NOT EXISTS ps DOUBLE PRECISION",
        "ALTER TABLE daily_basic ADD COLUMN IF NOT EXISTS ps_ttm DOUBLE PRECISION",
        "ALTER TABLE daily_basic ADD COLUMN IF NOT EXISTS dv_ratio DOUBLE PRECISION",
        "ALTER TABLE daily_basic ADD COLUMN IF NOT EXISTS dv_ttm DOUBLE PRECISION",
        "ALTER TABLE daily_basic ADD COLUMN IF NOT EXISTS total_share DOUBLE PRECISION",
        "ALTER TABLE daily_basic ADD COLUMN IF NOT EXISTS float_share DOUBLE PRECISION",
        "ALTER TABLE daily_basic ADD COLUMN IF NOT EXISTS free_share DOUBLE PRECISION",
        "ALTER TABLE moneyflow_daily ADD COLUMN IF NOT EXISTS buy_sm_amount DOUBLE PRECISION",
        "ALTER TABLE moneyflow_daily ADD COLUMN IF NOT EXISTS sell_sm_amount DOUBLE PRECISION",
        "ALTER TABLE moneyflow_daily ADD COLUMN IF NOT EXISTS buy_md_amount DOUBLE PRECISION",
        "ALTER TABLE moneyflow_daily ADD COLUMN IF NOT EXISTS sell_md_amount DOUBLE PRECISION",
    ]
    engine = get_engine()
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


@contextmanager
def session_scope() -> Session:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
