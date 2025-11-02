#!/usr/bin/env python3
"""
简易回测启动脚本（使用 TradingAgents-CN 内置 backtest 引擎）

示例：
  DATA_SOURCE=akshare MYTRADE_FAST_SIGNAL=true \
  python scripts/run_backtest.py --symbols 600519 000858 --start 2024-01-01 --end 2024-06-30
"""

import os
import argparse
from tradingagents.backtest.backtest_engine import BacktestEngine, BacktestConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='+', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--cash', type=float, default=1_000_000)
    ap.add_argument('--exec', dest='exec_rule', default='next_open', choices=['next_open','next_close','vwap'])
    args = ap.parse_args()

    cfg = BacktestConfig(symbols=args.symbols, start_date=args.start, end_date=args.end,
                         initial_cash=args.cash, exec_rule=args.exec_rule)
    eng = BacktestEngine()
    res = eng.run(cfg)
    print('Backtest done. Output:', res.output_dir)


if __name__ == '__main__':
    # 默认开启快速信号（无需LLM）
    os.environ.setdefault('MYTRADE_FAST_SIGNAL', 'true')
    main()

