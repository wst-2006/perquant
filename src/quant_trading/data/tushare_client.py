from __future__ import annotations

import time
from functools import wraps
from typing import Callable, TypeVar

import pandas as pd
import tushare as ts

from ..config import settings

T = TypeVar("T")


FINA_INDICATOR_FACTOR_NAMES = (
    "roe",
    "roe_dt",
    "roe_waa",
    "roa",
    "roic",
    "grossprofit_margin",
    "netprofit_margin",
    "ocf_to_profit",
    "debt_to_assets",
    "current_ratio",
    "quick_ratio",
    "netprofit_yoy",
    "dt_netprofit_yoy",
    "tr_yoy",
    "or_yoy",
    "ocf_yoy",
    "q_roe",
    "q_sales_yoy",
    "q_profit_yoy",
    "q_profit_qoq",
    "q_op_qoq",
    "q_gsprofit_margin",
    "q_netprofit_margin",
    "assets_yoy",
    "eqt_yoy",
    "rd_exp",
)
FINA_INDICATOR_STRUCTURALLY_SPARSE_FIELDS = ("ocf_to_profit", "rd_exp")
FINA_INDICATOR_GATE_FIELDS = tuple(
    name for name in FINA_INDICATOR_FACTOR_NAMES if name not in FINA_INDICATOR_STRUCTURALLY_SPARSE_FIELDS
)
FINA_INDICATOR_FACTOR_FIELDS = ",".join(("ts_code", "ann_date", "end_date", *FINA_INDICATOR_FACTOR_NAMES))


class TushareDataError(RuntimeError):
    """Raised when a required Tushare call fails."""


def _retry(fn: Callable[..., T]) -> Callable[..., T]:
    @wraps(fn)
    def wrapper(self: "TushareClient", *args, **kwargs):
        last_error: Exception | None = None
        for attempt in range(1, self.retry_times + 1):
            try:
                self._sleep()
                return fn(self, *args, **kwargs)
            except Exception as exc:  # Tushare raises heterogeneous exceptions
                last_error = exc
                if attempt < self.retry_times:
                    time.sleep(self.request_delay_seconds * attempt)
        raise TushareDataError(f"Tushare call failed: {fn.__name__}: {last_error}") from last_error

    return wrapper


class TushareClient:
    """Thin wrapper around Tushare Pro.

    Core methods never fabricate data. Empty DataFrames are returned only when the
    upstream interface returns no rows for the requested real query.
    """

    def __init__(self, token: str | None = None):
        self.token = token or settings.tushare.require_token()
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        self.request_delay_seconds = settings.tushare.request_delay_seconds
        self.retry_times = settings.tushare.retry_times

    def _sleep(self) -> None:
        if self.request_delay_seconds > 0:
            time.sleep(self.request_delay_seconds)

    @_retry
    def get_stock_basic(self, list_status: str = "L", with_filter: bool = False) -> pd.DataFrame:
        df = self.pro.stock_basic(
            exchange="",
            list_status=list_status,
            fields="ts_code,symbol,name,area,industry,market,list_date,is_hs",
        )
        if df is None:
            return pd.DataFrame()
        if with_filter and not df.empty:
            df = df[~df["name"].astype(str).str.contains("ST|退", regex=True, na=False)].copy()
        if not df.empty:
            df["list_status"] = list_status
        return df

    @_retry
    def get_all_a_shares(self, with_filter: bool = False) -> pd.DataFrame:
        return self.get_stock_basic(list_status="L", with_filter=with_filter)

    @_retry
    def get_full_stock_universe(self) -> pd.DataFrame:
        frames = [self.get_stock_basic(list_status=status, with_filter=False) for status in ("L", "D", "P")]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates(subset=["ts_code"], keep="first")

    @_retry
    def get_daily_quotes(self, ts_code: str | None = None, start_date: str | None = None, end_date: str | None = None, trade_date: str | None = None) -> pd.DataFrame:
        kwargs: dict[str, str] = {}
        if ts_code:
            kwargs["ts_code"] = ts_code
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        if trade_date:
            kwargs["trade_date"] = trade_date
        df = self.pro.daily(**kwargs)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_daily_basic(self, trade_date: str | None = None, ts_code: str | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        kwargs: dict[str, str] = {}
        if trade_date:
            kwargs["trade_date"] = trade_date
        if ts_code:
            kwargs["ts_code"] = ts_code
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        fields = "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
        df = self.pro.daily_basic(fields=fields, **kwargs)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_moneyflow(self, ts_code: str | None = None, start_date: str | None = None, end_date: str | None = None, trade_date: str | None = None) -> pd.DataFrame:
        kwargs: dict[str, str] = {}
        if ts_code:
            kwargs["ts_code"] = ts_code
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        if trade_date:
            kwargs["trade_date"] = trade_date
        df = self.pro.moneyflow(**kwargs)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_adj_factor(self, ts_code: str | None = None, start_date: str | None = None, end_date: str | None = None, trade_date: str | None = None) -> pd.DataFrame:
        kwargs: dict[str, str] = {}
        if ts_code:
            kwargs["ts_code"] = ts_code
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        if trade_date:
            kwargs["trade_date"] = trade_date
        df = self.pro.adj_factor(**kwargs)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_stk_limit(self, trade_date: str) -> pd.DataFrame:
        df = self.pro.stk_limit(trade_date=trade_date)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_trade_calendar(self, start_date: str, end_date: str, exchange: str = "SSE") -> pd.DataFrame:
        df = self.pro.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date, is_open="1")
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_top_list(self, trade_date: str) -> pd.DataFrame:
        df = self.pro.top_list(trade_date=trade_date)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_top_inst(self, trade_date: str) -> pd.DataFrame:
        df = self.pro.top_inst(trade_date=trade_date)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_hsgt_top10(self, trade_date: str | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        kwargs: dict[str, str] = {}
        if trade_date:
            kwargs["trade_date"] = trade_date
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        df = self.pro.hsgt_top10(**kwargs)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_moneyflow_hsgt(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.pro.moneyflow_hsgt(start_date=start_date, end_date=end_date)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_limit_list(self, trade_date: str, limit_type: str | None = None) -> pd.DataFrame:
        kwargs: dict[str, str] = {"trade_date": trade_date}
        if limit_type:
            kwargs["limit_type"] = limit_type
        df = self.pro.query("limit_list_d", **kwargs)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_suspend_daily(self, trade_date: str) -> pd.DataFrame:
        df = self.pro.query("suspend_d", trade_date=trade_date)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_v9_event_range(
        self,
        endpoint: str,
        start_date: str,
        end_date: str,
        *,
        limit: int = 6000,
        offset: int = 0,
    ) -> pd.DataFrame:
        df = self.pro.query(endpoint, start_date=start_date, end_date=end_date, limit=limit, offset=offset)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df if df is not None else pd.DataFrame()

    @_retry
    def query_endpoint(self, endpoint: str, fields: str | None = None, **kwargs) -> pd.DataFrame:
        cleaned = {key: value for key, value in kwargs.items() if value not in (None, "")}
        if fields:
            cleaned["fields"] = fields
        df = self.pro.query(endpoint, **cleaned)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_financial_endpoint(
        self,
        endpoint: str,
        ts_code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str | None = None,
        ann_date: str | None = None,
        fields: str | None = None,
        **extra,
    ) -> pd.DataFrame:
        kwargs: dict[str, str] = {}
        if ts_code:
            kwargs["ts_code"] = ts_code
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        if period:
            kwargs["period"] = period
        if ann_date:
            kwargs["ann_date"] = ann_date
        kwargs.update({key: value for key, value in extra.items() if value not in (None, "")})
        effective_fields = fields
        if effective_fields is None and endpoint in {"fina_indicator", "fina_indicator_vip"}:
            effective_fields = FINA_INDICATOR_FACTOR_FIELDS
        return self.query_endpoint(endpoint, fields=effective_fields, **kwargs)

    @_retry
    def get_index_basic(self, market: str = "SW") -> pd.DataFrame:
        df = self.pro.index_basic(market=market)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_index_member(self, index_code: str | None = None, ts_code: str | None = None) -> pd.DataFrame:
        kwargs: dict[str, str] = {}
        if index_code:
            kwargs["index_code"] = index_code
        if ts_code:
            kwargs["ts_code"] = ts_code
        df = self.pro.index_member(**kwargs)
        return df if df is not None else pd.DataFrame()

    @_retry
    def get_fina_indicator(self, ts_code: str | None = None, start_date: str | None = None, end_date: str | None = None, period: str | None = None) -> pd.DataFrame:
        kwargs: dict[str, str] = {}
        if ts_code:
            kwargs["ts_code"] = ts_code
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        if period:
            kwargs["period"] = period
        fields = "ts_code,ann_date,end_date,eps,dt_eps,total_revenue_ps,revenue_ps,capital_rese_ps,surplus_rese_ps,undist_profit_ps,extra_item,profit_dedt,gross_margin,current_ratio,quick_ratio,cash_ratio,ar_turn,ca_turn,fa_turn,assets_turn,op_income,valuechange_income,interst_income,daa,ebit,ebitda,fcff,fcfe,current_exint,noncurrent_exint,interestdebt,netdebt,tangible_asset,working_capital,networking_capital,invest_capital,retained_earnings,diluted2_eps,bps,ocfps,retainedps,cfps,ebit_ps,fcff_ps,fcfe_ps,netprofit_margin,grossprofit_margin,cogs_of_sales,expense_of_sales,profit_to_gr,saleexp_to_gr,adminexp_of_gr,finaexp_of_gr,impai_ttm,gc_of_gr,op_of_gr,ebit_of_gr,roe,roe_waa,roe_dt,roa,npta,roic,roe_yearly,roa2_yearly,roe_avg,opincome_of_ebt,investincome_of_ebt,n_op_profit_of_ebt,tax_to_ebt,dtprofit_to_profit,salescash_to_or,ocf_to_or,ocf_to_opincome,capitalized_to_da,debt_to_assets,assets_to_eqt,dp_assets_to_eqt,ca_to_assets,nca_to_assets,tbassets_to_totalassets,int_to_talcap,eqt_to_talcapital,currentdebt_to_debt,longdeb_to_debt,ocf_to_shortdebt,debt_to_eqt,eqt_to_debt,eqt_to_interestdebt,tangibleasset_to_debt,tangasset_to_intdebt,tangibleasset_to_netdebt,ocf_to_debtturn,ocf_to_interestdebt,ocf_to_netdebt,ebit_to_interest,longdebt_to_workingcapital,ebitda_to_debt,turn_days,roa_yearly,roa_dp,fix_assets,profit_prefin_exp,non_op_profit,op_to_ebt,nop_to_ebt,ocf_to_profit,cash_to_liqdebt,cash_to_liqdebt_withinterest,op_to_liqdebt,op_to_debt,roic_yearly,total_fa_trun,profit_to_op,q_opincome,q_investincome,q_dtprofit,q_eps,q_netprofit_margin,q_gsprofit_margin,q_exp_to_sales,q_profit_to_gr,q_saleexp_to_gr,q_adminexp_to_gr,q_finaexp_to_gr,q_impair_to_gr_ttm,q_gc_to_gr,q_op_to_gr,q_roe,q_dt_roe,q_npta,q_opincome_to_ebt,q_investincome_to_ebt,q_dtprofit_to_profit,q_salescash_to_or,q_ocf_to_sales,q_ocf_to_or,basic_eps_yoy,dt_eps_yoy,cfps_yoy,op_yoy,ebt_yoy,netprofit_yoy,dt_netprofit_yoy,ocf_yoy,roe_yoy,bps_yoy,assets_yoy,eqt_yoy,tr_yoy,or_yoy,q_gr_yoy,q_gr_qoq,q_sales_yoy,q_sales_qoq,q_op_yoy,q_op_qoq,q_profit_yoy,q_profit_qoq,q_netprofit_yoy,q_netprofit_qoq,equity_yoy,rd_exp"
        df = self.pro.fina_indicator(fields=fields, **kwargs)
        return df if df is not None else pd.DataFrame()
