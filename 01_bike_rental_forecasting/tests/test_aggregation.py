"""Tests for hourly rental aggregation logic."""

import pandas as pd

from bike_rental.defs.preprocessing.aggregation import aggregate_hourly_rentals


def test_aggregate_hourly_rentals_counts_booked_direct_and_total():
    """Test hourly aggregation counts for booked and direct rentals."""
    operational_rentals = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 10:05",
                    "2024-01-01 10:20",
                    "2024-01-01 10:45",
                    "2024-01-01 11:10",
                ]
            ),
            "datetime_hour": pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-01 10:00",
                    "2024-01-01 10:00",
                    "2024-01-01 11:00",
                ]
            ),
            "location_id": [100, 100, 100, 100],
            "is_booked": [1, 0, 1, 0],
            "source_dataset": [
                "booked_rentals",
                "direct_pickups",
                "booked_rentals",
                "direct_pickups",
            ],
        }
    )

    result = aggregate_hourly_rentals(operational_rentals)

    first_hour = result[result["datetime_hour"] == pd.Timestamp("2024-01-01 10:00")].iloc[0]

    assert first_hour["booked_rentals"] == 2
    assert first_hour["direct_pickups"] == 1
    assert first_hour["total_rentals"] == 3
