"""Asset checks for raw source datasets."""

import pandas as pd
from dagster import AssetCheckResult, asset_check

from bike_rental.defs.assets.preprocessing import hourly_rental_activity
from bike_rental.defs.assets.sources import (
    booked_rentals,
    direct_pickups,
    holidays,
    weather,
)
from bike_rental.defs.constants import (
    BOOKED_RENTALS_COLUMNS,
    DIRECT_PICKUPS_COLUMNS,
    HOLIDAYS_COLUMNS,
    WEATHER_COLUMNS,
)


def _check_columns_exist(
    df: pd.DataFrame,
    required_columns: set[str],
) -> AssetCheckResult:
    """Verify that all required columns exist."""
    missing_columns = required_columns - set(df.columns)

    return AssetCheckResult(
        passed=not missing_columns,
        metadata={
            "required_columns": sorted(required_columns),
            "missing_columns": sorted(missing_columns),
            "column_count": len(df.columns),
        },
    )


@asset_check(asset=booked_rentals)
def booked_rentals_schema(
    booked_rentals: pd.DataFrame,
) -> AssetCheckResult:
    """Check that the booked rentals dataset contains the expected columns."""
    return _check_columns_exist(
        booked_rentals,
        BOOKED_RENTALS_COLUMNS,
    )


@asset_check(asset=direct_pickups)
def direct_pickups_schema(
    direct_pickups: pd.DataFrame,
) -> AssetCheckResult:
    """Check that the direct pickups dataset contains the expected columns."""
    return _check_columns_exist(
        direct_pickups,
        DIRECT_PICKUPS_COLUMNS,
    )


@asset_check(asset=weather)
def weather_schema(
    weather: pd.DataFrame,
) -> AssetCheckResult:
    """Check that the weather dataset contains the expected columns."""
    return _check_columns_exist(
        weather,
        WEATHER_COLUMNS,
    )


@asset_check(asset=holidays)
def holidays_schema(
    holidays: pd.DataFrame,
) -> AssetCheckResult:
    """Check that the holiday calendar dataset contains the expected columns."""
    return _check_columns_exist(
        holidays,
        HOLIDAYS_COLUMNS,
    )


@asset_check(asset=hourly_rental_activity)
def hourly_rental_activity_has_continuous_hours(
    hourly_rental_activity: pd.DataFrame,
) -> AssetCheckResult:
    """Check that the hourly rental activity timeline contains no missing hours."""
    expected_hours = pd.date_range(
        start=hourly_rental_activity["datetime_hour"].min(),
        end=hourly_rental_activity["datetime_hour"].max(),
        freq="h",
    )

    passed = len(expected_hours) == len(hourly_rental_activity)

    return AssetCheckResult(
        passed=passed,
        metadata={
            "expected_hours": len(expected_hours),
            "actual_hours": len(hourly_rental_activity),
        },
    )
