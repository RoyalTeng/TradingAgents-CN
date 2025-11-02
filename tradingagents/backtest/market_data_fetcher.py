"""
市场数据采集器（简化版）

从多数据源获取A股历史行情数据，并带本地缓存。
此实现来自 myTrade 项目，做了最小改动以在 TradingAgents-CN 中独立使用。
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Literal, Union, List

import pandas as pd
from pydantic import BaseModel, Field


class DataSourceConfig(BaseModel):
    source: Literal["akshare", "tushare", "baostock", "mootdx", "rqdatac"] = "akshare"
    tushare_token: Optional[str] = None
    cache_dir: Union[Path, str] = Field(default="./data/cache")
    cache_days: int = 7
    enable_fallback: bool = True
    # 可选数据源配置（保留字段以便后续扩展）
    tdx_dir: Optional[Union[str, Path]] = None
    tdx_mode: Literal["offline", "online", "auto"] = "auto"
    tdx_servers: Optional[List[tuple]] = None
    tdx_page_size: int = 800
    tdx_retry: int = 2
    tdx_timeout: float = 15.0
    volume_unit: Literal["hand", "share"] = "hand"
    rq_user: Optional[str] = None
    rq_password: Optional[str] = None


class MarketDataFetcher:
    """简化的行情数据抓取与缓存器"""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache_dir = Path(config.cache_dir) if isinstance(config.cache_dir, str) else config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # lazy setup for tushare
        self.tushare_token = config.tushare_token or os.getenv('TUSHARE_TOKEN')
        self.ts_pro = None
        if config.source == "tushare" and self.tushare_token:
            try:
                import tushare as ts  # type: ignore
                ts.set_token(self.tushare_token)
                self.ts_pro = ts.pro_api()
            except Exception as e:
                self.logger.warning(f"Init Tushare failed: {e}")
                self.ts_pro = None

    # -------------------- Public API --------------------
    def fetch_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        freq: Literal["daily", "1min", "5min", "15min", "30min", "60min"] = "daily",
        force_update: bool = False,
    ) -> pd.DataFrame:
        """获取标准化后的 OHLCV DataFrame，索引为 DatetimeIndex。
        列: open, high, low, close, volume
        """
        norm_symbol = self._normalize_symbol(symbol)
        cache_file = self._cache_path(norm_symbol, freq)

        if not force_update and cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                mask = (df.index >= start_date) & (df.index <= end_date)
                return df[mask].copy()
            except Exception as e:
                self.logger.debug(f"read cache failed, refetch: {e}")

        # try sources with fallback
        for source in self._candidate_sources():
            try:
                df = self._fetch_from(source, norm_symbol, start_date, end_date, freq)
                if df is None or df.empty:
                    continue
                df = self._standardize(df)
                df.to_csv(cache_file)
                return df
            except Exception as e:
                self.logger.debug(f"source {source} failed: {e}")

        raise RuntimeError(f"no data for {symbol} {start_date}~{end_date}")

    # -------------------- Internal helpers --------------------
    def _candidate_sources(self) -> List[str]:
        base = [self.config.source]
        fallback = [s for s in ["rqdatac", "akshare", "tushare", "baostock"] if s not in base]
        return base + (fallback if self.config.enable_fallback else [])

    def _fetch_from(self, source: str, symbol: str, start: str, end: str, freq: str) -> Optional[pd.DataFrame]:
        if source == "akshare":
            return self._fetch_from_akshare(symbol, start, end, freq)
        if source == "tushare":
            return self._fetch_from_tushare(symbol, start, end, freq)
        if source == "baostock":
            return self._fetch_from_baostock(symbol, start, end, freq)
        if source == "rqdatac":
            return self._fetch_from_rqdatac(symbol, start, end, freq)
        raise ValueError(source)

    def _fetch_from_akshare(self, symbol: str, start: str, end: str, freq: str) -> Optional[pd.DataFrame]:
        try:
            import akshare as ak  # type: ignore
            if freq != "daily":
                raise NotImplementedError("akshare: only daily supported in simplified fetcher")
            s = symbol.replace('.SH', '').replace('.SZ', '')
            df = ak.stock_zh_a_hist(symbol=s, period="daily",
                                    start_date=start.replace('-', ''), end_date=end.replace('-', ''), adjust="")
            return df
        except Exception as e:
            self.logger.debug(f"akshare failed: {e}")
            return None

    def _fetch_from_tushare(self, symbol: str, start: str, end: str, freq: str) -> Optional[pd.DataFrame]:
        try:
            if not self.ts_pro:
                return None
            ts_code = self._to_ts_code(symbol)
            if freq != "daily":
                freq_map = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "60min": "60min"}
                ktype = freq_map.get(freq, None)
                if not ktype:
                    return None
                import tushare as ts  # type: ignore
                df = ts.pro_bar(ts_code=ts_code, start_date=start.replace('-', ''), end_date=end.replace('-', ''), freq=ktype)
            else:
                df = self.ts_pro.daily(ts_code=ts_code, start_date=start.replace('-', ''), end_date=end.replace('-', ''))
            return df
        except Exception as e:
            self.logger.debug(f"tushare failed: {e}")
            return None

    def _fetch_from_baostock(self, symbol: str, start: str, end: str, freq: str) -> Optional[pd.DataFrame]:
        try:
            import baostock as bs  # type: ignore
            lg = bs.login()
            if lg.error_code != '0':
                return None
            fields = "date,code,open,high,low,close,volume"
            rs = bs.query_history_k_data_plus(self._to_bs_code(symbol), fields, start_date=start, end_date=end, frequency='d', adjustflag='3')
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            bs.logout()
            if not data_list:
                return None
            df = pd.DataFrame(data_list, columns=fields.split(','))
            return df
        except Exception as e:
            self.logger.debug(f"baostock failed: {e}")
            return None

    def _fetch_from_rqdatac(self, symbol: str, start: str, end: str, freq: str) -> Optional[pd.DataFrame]:
        try:
            import rqdatac as rq  # type: ignore
            try:
                rq.init()
            except Exception:
                pass
            df = rq.get_price(self._to_rq_code(symbol), start_date=start, end_date=end, frequency=freq if freq != 'daily' else '1d')
            return df
        except Exception as e:
            self.logger.debug(f"rqdatac failed: {e}")
            return None

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        # 尝试统一列名到 open/high/low/close/volume
        mapping_candidates = [
            {"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "日期": "date"},
            {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "date": "date", "trade_date": "date"},
            {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Date": "date"},
        ]
        cols = {c: c for c in df.columns}
        for cand in mapping_candidates:
            for k, v in cand.items():
                if k in df.columns:
                    cols[k] = v
        df = df.rename(columns=cols)
        # 索引
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
        df = df.sort_index()
        # 仅保留需要的列
        keep = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        return df[keep].copy()

    def _normalize_symbol(self, s: str) -> str:
        s = s.upper()
        if s.endswith('.SH') or s.endswith('.SZ'):
            return s
        # 简化：以 6 开头当作上交所，其余深交所（业务可自行改进）
        return (s + ('.SH' if s.startswith('6') else '.SZ')) if len(s) == 6 else s

    def _to_ts_code(self, s: str) -> str:
        ns = self._normalize_symbol(s)
        return ns.replace('.SH', '.SH').replace('.SZ', '.SZ')

    def _to_bs_code(self, s: str) -> str:
        ns = self._normalize_symbol(s)
        return ('sh.' + ns[:6]) if ns.endswith('.SH') else ('sz.' + ns[:6])

    def _to_rq_code(self, s: str) -> str:
        ns = self._normalize_symbol(s)
        return ns

    def _cache_path(self, symbol: str, freq: str) -> Path:
        sub = self.cache_dir / f"{symbol.replace('.', '_')}_{freq}.csv"
        sub.parent.mkdir(parents=True, exist_ok=True)
        return sub

