"""Tests for the prediction API feature builder."""

import numpy as np
import pandas as pd

from bike_rental.api.features import build_forecast_features
from bike_rental.api.schemas import HourlyWeather

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity",
    "weekday",
    "is_weekend",
    "is_holiday",
    "conditions_clouds",
    "conditions_heavy_rain",
    "conditions_light_rain",
    "hour_sin",
    "hour_cos",
    "lag_24h",
    "lag_168h",
    "same_hour_mean_7d",
    "same_weekday_hour_mean_4w",
]


def _history(days: int = 60) -> pd.DataFrame:
    """Synthetic pre-modeling history ending at hour 23."""
    n = days * 24
    idx = pd.date_range("2011-01-01", periods=n, freq="h")
    hour = idx.hour
    rng = np.random.default_rng(0)
    total = np.clip(50 + 30 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 8, n), 0, None)
    return pd.DataFrame(
        {
            "datetime_hour": idx,
            "hour": hour,
            "weekday": idx.weekday,
            "month": idx.month,
            "is_weekend": idx.weekday >= 5,
            "is_holiday": False,
            "conditions": "clear",
            "temperature_c": 15.0,
            "perceived_temperature_c": 14.0,
            "humidity": 70.0,
            "windspeed_kmh": 5.0,
            "total_rentals": total.round().astype(int),
        }
    )


def _weather(conditions: str = "clear") -> list[HourlyWeather]:
    return [
        HourlyWeather(
            conditions=conditions,
            temperature_c=18.0,
            perceived_temperature_c=17.0,
            humidity=65.0,
            windspeed_kmh=6.0,
        )
        for _ in range(24)
    ]


_NO_HOLIDAYS = pd.DataFrame({"date": pd.to_datetime([])})


def test_feature_matrix_is_24_rows_numeric_and_complete():
    """It returns 24 fully-numeric rows with exactly the requested columns."""
    matrix, index = build_forecast_features(_weather(), _history(), _NO_HOLIDAYS, FEATURE_COLUMNS)

    assert len(matrix) == 24
    assert len(index) == 24
    assert list(matrix.columns) == FEATURE_COLUMNS
    assert matrix.isna().sum().sum() == 0
    assert all(dtype == np.float64 for dtype in matrix.dtypes)


def test_missing_condition_columns_are_filled_with_zero():
    """It reindexes to the model columns, so unseen conditions become all-zero."""
    matrix, _ = build_forecast_features(
        _weather("clear"), _history(), _NO_HOLIDAYS, FEATURE_COLUMNS
    )

    # No request hour was rainy, so these columns must exist and be zero.
    assert (matrix["conditions_heavy_rain"] == 0).all()
    assert (matrix["conditions_light_rain"] == 0).all()


def test_lag_features_are_pulled_from_history():
    """It computes horizon lags from observed history, not the unknown target."""
    history = _history()
    matrix, index = build_forecast_features(_weather(), history, _NO_HOLIDAYS, FEATURE_COLUMNS)

    hour_8_position = int(np.where(index.hour == 8)[0][0])
    expected = history[history["datetime_hour"].dt.hour == 8].iloc[-1]["total_rentals"]

    assert matrix.loc[hour_8_position, "lag_24h"] == expected


def test_horizon_starts_immediately_after_history():
    """It anchors the forecast horizon to the last observed data point."""
    history = _history()
    _, index = build_forecast_features(_weather(), history, _NO_HOLIDAYS, FEATURE_COLUMNS)

    assert index[0] == history["datetime_hour"].max() + pd.Timedelta(hours=1)
