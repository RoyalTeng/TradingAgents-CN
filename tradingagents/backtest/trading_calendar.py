"""
A股交易日历（简化版）

从 myTrade 复制并最小化改动，以便在 TradingAgents-CN 中独立使用。
"""

from typing import List, Dict, Set, Optional, Tuple
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class MarketStatus(Enum):
    TRADING = "trading"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    SUSPENDED = "suspended"


class HolidayType(Enum):
    NEW_YEAR = "new_year"
    SPRING_FESTIVAL = "spring_festival"
    TOMB_SWEEPING = "tomb_sweeping"
    LABOR_DAY = "labor_day"
    DRAGON_BOAT = "dragon_boat"
    MID_AUTUMN = "mid_autumn"
    NATIONAL_DAY = "national_day"
    WEEKEND = "weekend"
    OTHER = "other"


@dataclass
class MarketHoliday:
    start_date: date
    end_date: date
    holiday_type: HolidayType
    name: str
    description: Optional[str] = None


@dataclass
class TradingSession:
    name: str
    start_time: str
    end_time: str
    def contains_time(self, t: str) -> bool:
        return self.start_time <= t <= self.end_time


class AShareTradingCalendar:
    TRADING_SESSIONS = [
        TradingSession("morning", "09:30", "11:30"),
        TradingSession("afternoon", "13:00", "15:00"),
    ]

    def __init__(self):
        self._holidays_cache: Dict[int, List[MarketHoliday]] = {}
        self._trading_days_cache: Dict[Tuple[date, date], Set[date]] = {}
        self._load_builtin_holidays()

    def _load_builtin_holidays(self):
        from datetime import date as _d
        h24 = [
            MarketHoliday(_d(2024, 1, 1), _d(2024, 1, 1), HolidayType.NEW_YEAR, "元旦"),
            MarketHoliday(_d(2024, 2, 10), _d(2024, 2, 17), HolidayType.SPRING_FESTIVAL, "春节"),
            MarketHoliday(_d(2024, 4, 4), _d(2024, 4, 6), HolidayType.TOMB_SWEEPING, "清明节"),
            MarketHoliday(_d(2024, 5, 1), _d(2024, 5, 5), HolidayType.LABOR_DAY, "劳动节"),
            MarketHoliday(_d(2024, 6, 10), _d(2024, 6, 10), HolidayType.DRAGON_BOAT, "端午节"),
            MarketHoliday(_d(2024, 9, 15), _d(2024, 9, 17), HolidayType.MID_AUTUMN, "中秋节"),
            MarketHoliday(_d(2024, 10, 1), _d(2024, 10, 7), HolidayType.NATIONAL_DAY, "国庆节"),
        ]
        h25 = [
            MarketHoliday(_d(2025, 1, 1), _d(2025, 1, 1), HolidayType.NEW_YEAR, "元旦"),
        ]
        self._holidays_cache[2024] = h24
        self._holidays_cache[2025] = h25

    def is_trading_day(self, d: date) -> bool:
        if d.weekday() >= 5:
            return False
        return not self.is_holiday(d)

    def is_holiday(self, d: date) -> bool:
        year = d.year
        for hd in self._holidays_cache.get(year, []):
            if hd.start_date <= d <= hd.end_date:
                return True
        return False

    def get_trading_days(self, start_date: date, end_date: date) -> List[date]:
        k = (start_date, end_date)
        if k in self._trading_days_cache:
            return sorted(list(self._trading_days_cache[k]))
        s: Set[date] = set()
        cur = start_date
        while cur <= end_date:
            if self.is_trading_day(cur):
                s.add(cur)
            cur += timedelta(days=1)
        self._trading_days_cache[k] = s
        return sorted(list(s))


def create_ashare_calendar() -> AShareTradingCalendar:
    return AShareTradingCalendar()

