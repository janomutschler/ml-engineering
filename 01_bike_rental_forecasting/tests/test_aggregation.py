"""Tests for rental activity aggregation transformations."""

import pandas as pd

from bike_rental.defs.preprocessing.aggregation import (
    aggregate_hourly_rental_activity,
)


def test_aggregate_hourly_rental_activity_counts_hourly_rentals():
    """It aggregates booked and direct rentals by hour."""
    booked_rentals = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 10:05:00",
                    "2024-01-01 10:45:00",
                    "2024-01-01 12:10:00",
                ]
            )
        }
    )

    direct_pickups = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 10:15:00",
                    "2024-01-01 11:20:00",
                ]
            )
        }
    )

    result = aggregate_hourly_rental_activity(
        booked_rentals=booked_rentals,
        direct_pickups=direct_pickups,
    )

    expected = pd.DataFrame(
        {
            "datetime_hour": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 11:00:00",
                    "2024-01-01 12:00:00",
                ]
            ),
            "booked_rentals": [2, 0, 1],
            "total_rentals": [3, 1, 1],
            "direct_pickups": [1, 1, 0],
        }
    )

    pd.testing.assert_frame_equal(result, expected)
