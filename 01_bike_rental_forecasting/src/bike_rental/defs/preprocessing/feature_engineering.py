"""Feature engineering functions for the bike rental preprocessing pipeline."""

import pandas as pd

from bike_rental.defs.preprocessing.validate import validate_required_columns

BASE_DATASET_COLUMNS = [
    "datetime_hour",
    "date",
    "location_id",
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "is_holiday",
    "holiday",
    "conditions",
    "temperature_c",
    "perceived_temperature_c",
    "humidity",
    "windspeed_kmh",
    "booked_rentals",
    "direct_pickups",
    "total_rentals",
]


def add_time_features(rentals: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based time features to the rental data."""
    validate_required_columns(
        rentals,
        {"datetime_hour", "location_id", "total_rentals"},
        "enriched rentals",
    )

    featured_rentals = rentals.copy()

    featured_rentals["hour"] = featured_rentals["datetime_hour"].dt.hour
    featured_rentals["weekday"] = featured_rentals["datetime_hour"].dt.weekday
    featured_rentals["month"] = featured_rentals["datetime_hour"].dt.month
    featured_rentals["is_weekend"] = featured_rentals["weekday"] >= 5

    return featured_rentals[BASE_DATASET_COLUMNS].reset_index(drop=True)
