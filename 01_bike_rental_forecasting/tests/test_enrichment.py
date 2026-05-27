"""Tests for weather and holiday enrichment logic."""

import pandas as pd

from bike_rental.defs.preprocessing.enrichment import (
    join_holiday_data,
    join_weather_data,
)


def test_join_weather_data_adds_weather_columns_without_changing_row_count():
    """Test that weather enrichment preserves row counts."""
    hourly_rentals = pd.DataFrame(
        {
            "datetime_hour": pd.to_datetime(["2024-01-01 10:00"]),
            "location_id": [100],
            "booked_rentals": [2],
            "direct_pickups": [1],
            "total_rentals": [3],
        }
    )
    weather = pd.DataFrame(
        {
            "id": [1],
            "datetime": pd.to_datetime(["2024-01-01 10:00"]),
            "datetime_hour": pd.to_datetime(["2024-01-01 10:00"]),
            "conditions": ["clear"],
            "temperature_c": [10.0],
            "perceived_temperature_c": [9.0],
            "humidity": [70],
            "windspeed_kmh": [5],
        }
    )

    result = join_weather_data(hourly_rentals, weather)

    assert len(result) == 1
    assert result.loc[0, "conditions"] == "clear"
    assert result.loc[0, "temperature_c"] == 10.0


def test_join_holiday_data_adds_holiday_flag():
    """Test that holiday enrichment adds holiday metadata."""
    rentals = pd.DataFrame(
        {
            "datetime_hour": pd.to_datetime(["2024-01-01 10:00"]),
            "location_id": [100],
            "booked_rentals": [2],
            "direct_pickups": [1],
            "total_rentals": [3],
        }
    )
    holidays = pd.DataFrame(
        {
            "id": [1],
            "date": [pd.to_datetime("2024-01-01").date()],
            "holiday": ["New Year"],
        }
    )

    result = join_holiday_data(rentals, holidays)

    assert result.loc[0, "is_holiday"]
    assert result.loc[0, "holiday"] == "New Year"
