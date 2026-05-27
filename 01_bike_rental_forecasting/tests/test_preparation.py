"""Tests for preprocessing preparation and quarantine logic."""

import pandas as pd

from bike_rental.defs.preprocessing.preparation import (
    prepare_holidays,
    prepare_operational_rentals,
    prepare_weather,
)


def test_prepare_operational_rentals_quarantines_invalid_rows():
    """Test that invalid operational rental rows are quarantined."""
    booked = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "datetime": ["2024-01-01 10:15", "2024-01-01 10:20", "invalid"],
            "location_id": [100, 100, 101],
        }
    )
    direct = pd.DataFrame(
        {
            "id": [10, 11],
            "datetime": ["2024-01-01 11:00", "2024-01-01 12:00"],
            "location_id": [200, -1],
        }
    )

    valid, invalid = prepare_operational_rentals(booked, direct)

    assert len(valid) == 1
    assert len(invalid) == 4
    assert "quarantine_reason" in invalid.columns
    assert invalid["quarantine_reason"].str.len().gt(0).all()


def test_prepare_weather_quarantines_duplicate_hours():
    """Test that duplicate weather hours are quarantined."""
    weather = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "datetime": [
                "2024-01-01 10:00",
                "2024-01-01 10:30",
                "2024-01-01 11:00",
            ],
            "conditions": ["clear", "cloudy", "rain"],
            "temperature_c": [10.0, 11.0, 9.0],
            "perceived_temperature_c": [9.0, 10.0, 8.0],
            "humidity": [70, 75, 80],
            "windspeed_kmh": [5, 6, 7],
        }
    )

    valid, invalid = prepare_weather(weather)

    assert len(valid) == 1
    assert len(invalid) == 2
    assert invalid["quarantine_reason"].str.contains("duplicate datetime_hour").all()


def test_prepare_holidays_quarantines_duplicate_dates():
    """Test that invalid and duplicate holiday dates are quarantined."""
    holidays = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2024-01-01", "2024-01-01", "invalid"],
            "holiday": ["New Year", "Duplicate New Year", "Broken Holiday"],
        }
    )

    valid, invalid = prepare_holidays(holidays)

    assert len(valid) == 0
    assert len(invalid) == 3
    assert "quarantine_reason" in invalid.columns
