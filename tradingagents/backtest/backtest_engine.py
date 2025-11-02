"""
回测引擎（集成到 TradingAgents-CN）

从 myTrade 复制并做最小改动：
- 依赖的 MarketDataFetcher、TradingCalendar、PortfolioManager 一并内置于 tradingagents.backtest 包。
- SignalGenerator 为可选；若设置环境变量 MYTRADE_FAST_SIGNAL=true，则走快速信号生成，不依赖 LLM。
"""

import os
import json
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from .portfolio_manager import PortfolioManager, TradeRecord
from .market_data_fetcher import MarketDataFetcher, DataSourceConfig
from .trading_calendar import create_ashare_calendar


@dataclass
class BacktestConfig:
    symbols: List[str]
    start_date: str
    end_date: str
    initial_cash: float = 1_000_000.0
    max_positions: int = 10
    position_size: float = 0.1
    exec_rule: str = "next_open"    # next_open|next_close|vwap
    commission_bps: int = 3
    slippage_bps: int = 5
    stamp_duty_bps: int = 10
    frequency: str = "daily"


@dataclass
class BacktestResult:
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    equity_curve: pd.DataFrame
    trades: List[Dict[str, Any]]
    run_id: str
    output_dir: str


class BacktestEngine:
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path

    def run(self, cfg: BacktestConfig) -> BacktestResult:
        run_id = self._run_id()
        self.logger.info(f"开始回测，运行ID: {run_id}")

        # 1. 组件
        fetcher = self._create_fetcher()
        calendar = create_ashare_calendar()
        fast_mode = os.getenv('MYTRADE_FAST_SIGNAL', '').lower() in {'1', 'true', 'yes'}
        signal_gen = None
        if not fast_mode:
            try:
                from tradingagents.backtest_optional.signal_generator import SignalGenerator  # 可选
                signal_gen = SignalGenerator(self.config_path or "config.yaml")
            except Exception as e:
                self.logger.warning(f"SignalGenerator 不可用，自动使用快速模式: {e}")
                fast_mode = True

        # 2. 数据
        self.logger.info("获取历史数据...")
        data = self._fetch_data(fetcher, cfg)

        # 3. 交易日
        trading_days = self._trading_days_union(data, cfg.start_date, cfg.end_date)
        self.logger.info(f"交易日数量: {len(trading_days)}")

        # 4. 组合
        portfolio = PortfolioManager(cash=cfg.initial_cash)

        # 5. 主循环
        order_buffer: Dict[pd.Timestamp, List] = {}
        for i, dt in enumerate(trading_days):
            # 生成信号
            for symbol in cfg.symbols:
                if symbol in data and dt in data[symbol].index:
                    try:
                        if fast_mode:
                            sig = self._fast_signal(symbol, data[symbol].loc[:dt])
                            signal_report = {
                                'symbol': symbol,
                                # 为避免前视偏差检查，信号时间标记为上一日
                                'as_of': str((dt - pd.Timedelta(days=1)).date()),
                                'signal': sig,
                                'final_decision': {
                                    'action': sig.get('action', 'HOLD'),
                                    'confidence': sig.get('confidence', 0.5)
                                }
                            }
                        else:
                            # 仅在提供可选实现时可用
                            signal_report = signal_gen.generate_signal(symbol=symbol, target_date=dt)  # type: ignore
                        self._assert_no_lookahead(signal_report, dt)
                        order_buffer.setdefault(dt, []).append((symbol, signal_report))
                    except Exception as e:
                        self.logger.warning(f"信号生成失败 {symbol}@{dt}: {e}")

            # 执行订单（T+1）
            if i + 1 < len(trading_days):
                exec_dt = trading_days[i + 1]
                orders = order_buffer.get(dt, [])
                if orders:
                    self._execute_orders(portfolio, orders, data, exec_dt, cfg)

            # 记净值
            closes = self._closes_on(data, cfg.symbols, dt)
            portfolio.mark_to_market(closes, dt.strftime('%Y-%m-%d'))

        # 6. 指标
        equity_curve = portfolio.get_equity_curve()
        metrics = self._metrics(equity_curve)

        # 7. 结果
        result = BacktestResult(
            config=asdict(cfg),
            metrics=metrics,
            equity_curve=equity_curve,
            trades=[t for t in portfolio.trades],
            run_id=run_id,
            output_dir=""
        )
        output_dir = self._write_artifacts(run_id, result)
        result.output_dir = output_dir
        self.logger.info(f"回测完成，结果保存在: {output_dir}")
        return result

    # ----------------- helpers -----------------
    def _run_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"backtest_{ts}_{str(uuid.uuid4())[:8]}"

    def _create_fetcher(self) -> MarketDataFetcher:
        cfg = DataSourceConfig(source=os.getenv('DATA_SOURCE', 'akshare'), cache_dir="data/cache", cache_days=7, enable_fallback=True)
        return MarketDataFetcher(cfg)

    def _fetch_data(self, fetcher: MarketDataFetcher, cfg: BacktestConfig) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for s in cfg.symbols:
            try:
                df = fetcher.fetch_history(s, cfg.start_date, cfg.end_date, cfg.frequency)
                if df is not None and not df.empty:
                    data[s] = df
                    self.logger.info(f"获取 {s} 数据: {len(df)} 条")
                else:
                    self.logger.warning(f"未获取到 {s} 的数据")
            except Exception as e:
                self.logger.error(f"获取 {s} 数据失败: {e}")
        if not data:
            raise ValueError("未能获取任何股票数据")
        return data

    def _trading_days_union(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> List[pd.Timestamp]:
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        return sorted([dt for dt in all_dates if start_dt <= dt <= end_dt])

    def _assert_no_lookahead(self, signal_report: Dict[str, Any], exec_dt: pd.Timestamp):
        as_of = signal_report.get('as_of')
        if as_of:
            sdt = pd.Timestamp(as_of)
            if sdt >= exec_dt:
                raise ValueError(f"前视偏差: 信号时间 {sdt} >= 执行时间 {exec_dt}")

    def _execute_orders(self, portfolio: PortfolioManager, orders: List, data: Dict[str, pd.DataFrame], exec_dt: pd.Timestamp, cfg: BacktestConfig):
        for symbol, signal_report in orders:
            if symbol not in data or exec_dt not in data[symbol].index:
                continue
            sig = signal_report['signal']
            action = sig.get('action', 'HOLD')
            weight = sig.get('weight', 0.0)
            if action == 'HOLD' or weight == 0:
                continue
            bar = data[symbol].loc[exec_dt]
            exec_price = self._exec_price(bar, action, cfg)
            position_cash = portfolio.cash * min(abs(weight), cfg.position_size)
            shares = int(position_cash / exec_price) if exec_price > 0 else 0
            if shares <= 0:
                continue
            commission = exec_price * shares * cfg.commission_bps / 10000
            stamp_duty = exec_price * shares * cfg.stamp_duty_bps / 10000 if action == 'SELL' else 0
            total_cost = commission + stamp_duty
            try:
                ok = False
                if action == 'BUY':
                    ok = portfolio.buy(symbol, shares, exec_price, exec_dt.strftime('%Y-%m-%d'), total_cost)
                elif action == 'SELL':
                    ok = portfolio.sell(symbol, shares, exec_price, exec_dt.strftime('%Y-%m-%d'), total_cost)
                if not ok:
                    self.logger.warning(f"交易执行失败: {action} {symbol} {shares}股")
            except Exception as e:
                self.logger.error(f"交易执行异常: {e}")

    def _exec_price(self, bar: pd.Series, action: str, cfg: BacktestConfig) -> float:
        if cfg.exec_rule == "next_open":
            base = bar['open']
        elif cfg.exec_rule == "next_close":
            base = bar['close']
        else:
            base = (bar['high'] + bar['low'] + bar['close']) / 3
        slip = base * cfg.slippage_bps / 10000
        return base + slip if action == 'BUY' else base - slip

    def _closes_on(self, data: Dict[str, pd.DataFrame], symbols: List[str], dt: pd.Timestamp) -> Dict[str, float]:
        closes: Dict[str, float] = {}
        for s in symbols:
            if s in data and dt in data[s].index:
                closes[s] = float(data[s].loc[dt, 'close'])
            else:
                closes[s] = self._fallback_price(data.get(s), dt)
        return closes

    def _fallback_price(self, df: Optional[pd.DataFrame], dt: pd.Timestamp) -> float:
        if df is None or df.empty:
            return 0.0
        avail = df.index[df.index <= dt]
        if len(avail) > 0:
            return float(df.loc[avail[-1], 'close'])
        return 0.0

    def _fast_signal(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            close = df['close'].iloc[-1]
            ma5 = df['close'].rolling(5).mean().iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else None
            if ma20 is not None and ma5 is not None and ma5 > ma20 * 1.002:
                action, score = 'BUY', 0.7
            elif ma20 is not None and ma5 is not None and ma5 < ma20 * 0.998:
                action, score = 'SELL', 0.3
            else:
                action, score = 'HOLD', 0.5
            weight = (score - 0.5) * 2
            return {
                'action': action,
                'confidence': 0.6,
                'weight': weight,
                'reasoning': [f"MA5={ma5:.2f} MA20={ma20:.2f} Close={close:.2f}" if ma5 is not None and ma20 is not None else '样本不足'],
                'rationale': '快速模式MA策略',
                'final_score': score,
            }
        except Exception as e:
            return {'action': 'HOLD', 'confidence': 0.1, 'weight': 0.0, 'reasoning': [f'快速信号失败: {e}'], 'rationale': '错误兜底'}

    def _metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {}
        equity = equity_curve['equity']
        rets = equity.pct_change().fillna(0.0)
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 0 else 0.0
        ann_return = (1 + rets.mean()) ** 252 - 1 if rets.mean() != 0 else 0.0
        ann_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min() if len(drawdown) > 0 else 0.0
        win_rate = float((rets > 0).sum() / len(rets)) if len(rets) > 0 else 0.0
        avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0.0
        avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0.0
        pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        return {
            'total_return': float(total_return),
            'annualized_return': float(ann_return),
            'annualized_volatility': float(ann_vol),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'profit_loss_ratio': float(pl_ratio),
            'trading_days': int(len(rets)),
        }

    def _write_artifacts(self, run_id: str, result: BacktestResult) -> str:
        out = Path("output") / "backtests" / run_id
        out.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(out / "equity_curve.csv")
        if result.trades:
            pd.DataFrame(result.trades).to_csv(out / "trades.csv", index=False)
        else:
            pd.DataFrame().to_csv(out / "trades.csv", index=False)
        with open(out / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(result.metrics, f, indent=2, ensure_ascii=False)
        report = (
            "# 回测报告\n\n"
            "## 基本信息\n"
            f"- 运行ID: {result.run_id}\n"
            f"- 回测期间: {result.config['start_date']} 至 {result.config['end_date']}\n"
            f"- 股票池: {', '.join(result.config['symbols'])}\n"
            f"- 初始资金: ¥{result.config['initial_cash']:,.2f}\n"
            f"- 执行规则: {result.config['exec_rule']}\n\n"
            "## 回测指标\n"
            f"- 总收益率: {result.metrics.get('total_return', 0)*100:.2f}%\n"
            f"- 年化收益率: {result.metrics.get('annualized_return', 0)*100:.2f}%\n"
            f"- 年化波动率: {result.metrics.get('annualized_volatility', 0)*100:.2f}%\n"
            f"- 夏普比率: {result.metrics.get('sharpe_ratio', 0):.2f}\n"
            f"- 最大回撤: {result.metrics.get('max_drawdown', 0)*100:.2f}%\n"
            f"- 胜率: {result.metrics.get('win_rate', 0)*100:.2f}%\n"
            f"- 盈亏比: {result.metrics.get('profit_loss_ratio', 0):.2f}\n"
            f"- 交易天数: {result.metrics.get('trading_days', 0)}\n\n"
            f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        )
        with open(out / "report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        return str(out)
