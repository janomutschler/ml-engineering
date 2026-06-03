"""Tests for calendar feature transformations."""

import pandas as pd

from bike_rental.defs.preprocessing.calendar_features import (
    create_calendar_features,
)


def test_create_calendar_features_adds_calendar_columns():
    """It creates calendar features from hourly rental activity."""
    hourly_rental_activity = pd.DataFrame(
        {
            "datetime_hour": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-06 11:00:00",
                ]
            )
        }
    )

    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                ]
            )
        }
    )

    result = create_calendar_features(
        hourly_rental_activity=hourly_rental_activity,
        holidays=holidays,
    )

    expected = pd.DataFrame(
        {
            "datetime_hour": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-06 11:00:00",
                ]
            ),
            "hour": [10, 11],
            "weekday": [0, 5],
            "month": [1, 1],
            "is_weekend": [False, True],
            "is_holiday": [True, False],
        }
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
    )
