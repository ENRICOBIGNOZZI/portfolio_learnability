import numpy as np
import pandas as pd


PANEL_META_COLUMNS = (
    "id",
    "eom",
    "permno",
    "me",
    "ret_exc_lead1m",
)


def as_characteristic_list(characteristics):
    """Return characteristic names as an ordered list."""
    if characteristics is None:
        return []
    if isinstance(characteristics, str):
        return [characteristics]
    return list(characteristics)


def rank_characteristics(values):
    """Rank a cross-section column-wise to the interval [-0.5, 0.5]."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("values must be a vector or a two-dimensional matrix.")
    if not np.isfinite(values).all():
        raise ValueError("values must contain only finite observations.")
    if len(values) <= 1:
        return np.zeros_like(values, dtype=float)

    ranks = pd.DataFrame(values).rank(method="average").to_numpy(dtype=float)
    return (ranks - 1.0) / (len(values) - 1.0) - 0.5


def compute_equity_line(returns, initial_value=1.0):
    """Turn a sequence of returns into an equity line."""
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1:
        raise ValueError("returns must be a one-dimensional vector.")
    if not np.isfinite(returns).all():
        raise ValueError("returns must contain only finite values.")
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")

    return initial_value * np.cumprod(
        1.0 + returns
    )


def compute_sharpe_ratio(returns, periods_per_year=12):
    """Compute the annualized Sharpe ratio."""
    returns = np.asarray(returns, dtype=float)

    if len(returns) < 2:
        return np.nan

    volatility = np.std(
        returns,
        ddof=1,
    )

    if volatility == 0:
        return np.nan

    return (
        np.sqrt(periods_per_year)
        * np.mean(returns)
        / volatility
    )


def limit_gross_exposure(weights, maximum=None):
    """Optionally limit gross exposure; None leaves weights uncapped."""
    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError("weights must be a one-dimensional vector.")
    if not np.isfinite(weights).all():
        raise ValueError("weights must contain only finite values.")

    raw_gross_exposure = np.abs(weights).sum()
    if maximum is None:
        return weights.copy(), raw_gross_exposure, 1.0
    if maximum <= 0:
        raise ValueError("maximum must be positive.")

    scale_factor = min(1.0, maximum / max(raw_gross_exposure, 1e-300))
    limited_weights = weights * scale_factor

    return limited_weights, raw_gross_exposure, scale_factor


def build_monthly_panels(
    df,
    characteristics=None,
    characteristics_are_ranked=None,
):
    """Build one investable stock cross-section for every month.

    Missing ``permno`` values are retained because ``id`` is the panel key.
    Rows with a missing/non-finite future return or characteristic are removed.
    Clean JKP frames advertise that their characteristics are already ranked;
    in that case the stored arrays are reused rather than ranked a second time.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if isinstance(characteristics, str) and characteristics.lower() == "all":
        characteristics = [
            column for column in df.columns if column not in PANEL_META_COLUMNS
        ]
    else:
        characteristics = as_characteristic_list(characteristics)
    if len(characteristics) != len(set(characteristics)):
        raise ValueError("characteristics contains duplicate column names.")

    required = [*PANEL_META_COLUMNS, *characteristics]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing panel columns: {missing}")

    if characteristics_are_ranked is None:
        characteristics_are_ranked = bool(
            df.attrs.get("characteristics_rank_standardized", False)
        )

    panels = []
    for date, cross_section in df.groupby("eom", sort=True, observed=True):
        returns = cross_section["ret_exc_lead1m"].to_numpy(dtype=float)
        valid = np.isfinite(returns)

        if characteristics:
            raw_characteristics = cross_section[characteristics].to_numpy(
                dtype=float
            )
            valid &= np.isfinite(raw_characteristics).all(axis=1)
        else:
            raw_characteristics = np.empty((len(cross_section), 0), dtype=float)

        if not valid.any():
            continue
        if not valid.all():
            cross_section = cross_section.loc[valid]
            returns = returns[valid]
            raw_characteristics = raw_characteristics[valid]

        if characteristics_are_ranked:
            ranked_characteristics = raw_characteristics
        else:
            ranked_characteristics = rank_characteristics(raw_characteristics)

        panels.append(
            {
                "date": pd.Timestamp(date),
                "formation_date": pd.Timestamp(date),
                "return_date": pd.Timestamp(date) + pd.offsets.MonthEnd(1),
                "id": cross_section["id"].to_numpy(),
                "permno": cross_section["permno"].to_numpy(),
                "me": cross_section["me"].to_numpy(dtype=float),
                "n_assets": len(cross_section),
                "characteristics": characteristics,
                "x_raw": raw_characteristics,
                "x": ranked_characteristics,
                "z": ranked_characteristics,
                "r": returns,
            }
        )

    return panels


def build_expanding_windows(
    panels,
    *,
    initial_train_years=10,
    validation_years=5,
    test_years=1,
):
    """Split monthly panels into expanding-train, fixed-length OOS windows.

    The training sample always begins in the first available year and expands
    by one year at each iteration. Validation and test move forward while
    retaining their requested lengths. Panel dictionaries and their arrays are
    referenced, not copied.
    """
    lengths = {
        "initial_train_years": initial_train_years,
        "validation_years": validation_years,
        "test_years": test_years,
    }
    for name, value in lengths.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    panels = sorted(
        list(panels),
        key=lambda panel: pd.Timestamp(panel["formation_date"]),
    )
    if not panels:
        return []

    dates = [pd.Timestamp(panel["formation_date"]) for panel in panels]
    if len(dates) != len(set(dates)):
        raise ValueError("panels must contain at most one cross-section per month.")

    panels_by_year = {}
    for panel, date in zip(panels, dates, strict=True):
        panels_by_year.setdefault(date.year, []).append(panel)

    first_year = min(panels_by_year)
    last_year = max(panels_by_year)
    first_test_year = first_year + initial_train_years + validation_years
    last_test_start = last_year - test_years + 1
    windows = []

    for test_start_year in range(first_test_year, last_test_start + 1):
        validation_start_year = test_start_year - validation_years
        train_year_range = tuple(range(first_year, validation_start_year))
        validation_year_range = tuple(
            range(validation_start_year, test_start_year)
        )
        test_year_range = tuple(
            range(test_start_year, test_start_year + test_years)
        )
        required_years = (
            *train_year_range,
            *validation_year_range,
            *test_year_range,
        )
        if any(year not in panels_by_year for year in required_years):
            continue

        windows.append(
            {
                "train_years": train_year_range,
                "validation_years": validation_year_range,
                "test_years": test_year_range,
                "train": [
                    panel
                    for year in train_year_range
                    for panel in panels_by_year[year]
                ],
                "validation": [
                    panel
                    for year in validation_year_range
                    for panel in panels_by_year[year]
                ],
                "test": [
                    panel
                    for year in test_year_range
                    for panel in panels_by_year[year]
                ],
            }
        )

    return windows
