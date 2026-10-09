"""Calendar-month chronology, never row-offset chronology."""
import numpy as np
import pandas as pd


def month_number(date):
    d = pd.Timestamp(date)
    return d.year * 12 + d.month


def next_month(date):
    return pd.Timestamp(date) + pd.offsets.MonthEnd(1)


def ages(origin, realization_dates, history_start):
    result = np.array([month_number(origin) + 1 - month_number(s)
                       for s in realization_dates], dtype=float)
    if len(result) == 0 or np.any(result < 1):
        raise ValueError('Historical payoffs must precede the next-month target.')
    span = month_number(origin) - month_number(history_start) + 1
    if span <= 0:
        raise ValueError('Invalid past-only calendar span.')
    return result, span


def eligible(panel, origin):
    return (panel.return_realization_date <= pd.Timestamp(origin)
            and panel.available_at <= pd.Timestamp(origin))
