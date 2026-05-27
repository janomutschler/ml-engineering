"""Tests for calendar-based feature engineering."""

import pandas as pd

from bike_rental.defs.preprocessing.feature_engineering import (
    BASE_DATASET_COLUMNS,
    add_time_features,
)


def test_add_time_features_creates_expected_calendar_features():
    """Test creation of calendar-based rental features."""
    rentals = pd.DataFrame(
        {
            "datetime_hour": pd.to_datetime(["2024-01-06 15:00"]),
            "date": [pd.to_datetime("2024-01-06").date()],
            "location_id": [100],
            "is_holiday": [False],
            "holiday": [None],
            "conditions": ["clear"],
            "temperature_c": [10.0],
            "perceived_temperature_c": [9.0],
            "humidity": [70],
            "windspeed_kmh": [5],
            "booked_rentals": [2],
            "direct_pickups": [1],
            "total_rentals": [3],
        }
    )

    result = add_time_features(rentals)

    assert result.loc[0, "hour"] == 15
    assert result.loc[0, "weekday"] == 5
    assert result.loc[0, "month"] == 1
    assert result.loc[0, "is_weekend"]
    assert list(result.columns) == BASE_DATASET_COLUMNS
