from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class DatabaseSettings:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "quant_db")
    user: str = os.getenv("DB_USER", "postgres")
    password: str | None = os.getenv("DB_PASSWORD")

    @property
    def url(self) -> str:
        if not self.password:
            raise ConfigError("DB_PASSWORD is missing. Create .env from .env.example.")
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass(frozen=True)
class TushareSettings:
    token: str | None = os.getenv("TUSHARE_TOKEN")
    points: int = int(os.getenv("TUSHARE_POINTS", "2200"))
    request_delay_seconds: float = float(os.getenv("TUSHARE_REQUEST_DELAY", "0.25"))
    retry_times: int = int(os.getenv("TUSHARE_RETRY_TIMES", "3"))

    def require_token(self) -> str:
        if not self.token:
            raise ConfigError("TUSHARE_TOKEN is missing. Create .env from .env.example.")
        return self.token


@dataclass(frozen=True)
class CapitalAllocation:
    """Optional reference split; selection ranking must not depend on it."""
    short_term_core: float = 0.60
    aggressive: float = 0.30
    cash: float = 0.10

    def validate(self) -> None:
        total = self.short_term_core + self.aggressive + self.cash
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(f"Capital allocation must sum to 1.0, got {total:.4f}.")


@dataclass(frozen=True)
class StrategySettings:
    total_capital: float | None = float(os.getenv("REFERENCE_CAPITAL")) if os.getenv("REFERENCE_CAPITAL") else None
    capital_mode: str = "manual"
    allocation_used_for_selection: bool = False
    allocation: CapitalAllocation = field(default_factory=CapitalAllocation)
    preliminary_filter_top_n: int = 2000
    stable_top_n: int = 10
    aggressive_top_n: int = 10
    min_score: float = 50.0

    factor_weights: dict[str, float] = field(
        default_factory=lambda: {
            "technical_score": 0.30,
            "capital_score": 0.25,
            "sector_strength": 0.15,
            "market_fit": 0.10,
            "valuation_score": 0.08,
            "growth_score": 0.07,
            "quality_score": 0.05,
        }
    )

    def validate(self) -> None:
        self.allocation.validate()
        weight_sum = sum(self.factor_weights.values())
        if abs(weight_sum - 1.0) > 1e-9:
            raise ConfigError(f"Factor weights must sum to 1.0, got {weight_sum:.4f}.")
        if self.preliminary_filter_top_n < 2000:
            raise ConfigError("Core selection must not shrink preliminary_filter_top_n below 2000.")


@dataclass(frozen=True)
class FlaskSettings:
    host: str = os.getenv("FLASK_HOST", "127.0.0.1")
    port: int = int(os.getenv("FLASK_PORT", "5000"))
    debug: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    secret_key: str = os.getenv("SECRET_KEY", "dev-only-change-me")


@dataclass(frozen=True)
class AISettings:
    enabled: bool = os.getenv("AI_ANALYSIS_ENABLED", "false").lower() == "true"
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY") or None


@dataclass(frozen=True)
class ResearchSettings:
    v8_artifact_path: str = os.getenv(
        "V8_ARTIFACT_PATH",
        "reports/ml_lab/v11_dual_artifact_feature_v2/artifact.json",
    )


@dataclass(frozen=True)
class Settings:
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    tushare: TushareSettings = field(default_factory=TushareSettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    flask: FlaskSettings = field(default_factory=FlaskSettings)
    ai: AISettings = field(default_factory=AISettings)
    research: ResearchSettings = field(default_factory=ResearchSettings)

    def validate_core(self) -> None:
        self.tushare.require_token()
        _ = self.database.url
        self.strategy.validate()

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name,
                "user": self.database.user,
                "password_configured": bool(self.database.password),
            },
            "tushare": {
                "token_configured": bool(self.tushare.token),
                "points": self.tushare.points,
            },
            "strategy": {
                "capital_mode": self.strategy.capital_mode,
                "reference_capital": self.strategy.total_capital,
                "allocation_used_for_selection": self.strategy.allocation_used_for_selection,
                "optional_reference_allocation": {
                    "short_term_core": self.strategy.allocation.short_term_core,
                    "aggressive": self.strategy.allocation.aggressive,
                    "cash": self.strategy.allocation.cash,
                },
                "preliminary_filter_top_n": self.strategy.preliminary_filter_top_n,
                "factor_weights": self.strategy.factor_weights,
            },
            "ai": {"enabled": self.ai.enabled},
            "research": {"v8_artifact_path": self.research.v8_artifact_path},
        }


settings = Settings()

