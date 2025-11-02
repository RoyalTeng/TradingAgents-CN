"""
投资组合管理器

管理投资组合的现金、持仓、交易记录等状态。
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd


@dataclass
class TradeRecord:
    """交易记录"""
    dt: str
    symbol: str
    side: str
    shares: int
    price: float
    cost: float


class PortfolioManager:
    """投资组合管理器"""
    
    def __init__(self, cash: float = 1_000_000.0):
        """初始化投资组合管理器
        
        Args:
            cash: 初始现金
        """
        self.cash = cash
        self.positions: Dict[str, Dict[str, float]] = {}
        self.trades: List[Dict[str, Any]] = []
        self._equity = []

    def buy(self, symbol: str, shares: int, price: float, dt: str, cost: float) -> bool:
        """买入股票"""
        pos = self.positions.setdefault(symbol, {"shares": 0, "avg_cost": 0.0})
        total = shares * price + cost
        
        if self.cash < total:
            return False
            
        new_shares = pos["shares"] + shares
        if new_shares > 0:
            pos["avg_cost"] = (pos["shares"] * pos["avg_cost"] + shares * price) / new_shares
        pos["shares"] = new_shares
        self.cash -= total
        
        trade_record = TradeRecord(dt, symbol, "BUY", shares, price, cost)
        self.trades.append(trade_record.__dict__)
        
        return True

    def sell(self, symbol: str, shares: int, price: float, dt: str, cost: float) -> bool:
        """卖出股票"""
        pos = self.positions.setdefault(symbol, {"shares": 0, "avg_cost": 0.0})
        actual_shares = min(shares, pos["shares"])
        
        if actual_shares <= 0:
            return False
            
        self.cash += actual_shares * price - cost
        pos["shares"] -= actual_shares
        
        if pos["shares"] == 0:
            pos["avg_cost"] = 0.0
            
        trade_record = TradeRecord(dt, symbol, "SELL", actual_shares, price, cost)
        self.trades.append(trade_record.__dict__)
        
        return True

    def mark_to_market(self, prices: Dict[str, float], dt: str) -> float:
        """按市价计算组合净值"""
        market_value = sum(
            pos["shares"] * prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )
        equity = self.cash + market_value
        self._equity.append({"dt": dt, "equity": equity})
        return equity

    def get_equity_curve(self) -> pd.DataFrame:
        """获取净值曲线"""
        if not self._equity:
            return pd.DataFrame(columns=["equity"])
        return pd.DataFrame(self._equity).set_index("dt")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取投资组合摘要"""
        total_shares_value = sum(
            pos["shares"] * pos["avg_cost"]
            for pos in self.positions.values()
            if pos["shares"] > 0
        )
        
        return {
            "cash": self.cash,
            "positions_count": len([p for p in self.positions.values() if p["shares"] > 0]),
            "total_shares_value": total_shares_value,
            "total_trades": len(self.trades),
            "positions": {k: v for k, v in self.positions.items() if v["shares"] > 0}
        }

