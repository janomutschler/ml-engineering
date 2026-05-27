"""Enrichment functions for the bike rental preprocessing pipeline."""

import pandas as pd

from bike_rental.defs.preprocessing.validate import validate_required_columns


def join_weather_data(
    hourly_rentals: pd.DataFrame,
    prepared_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Join hourly weather features onto hourly rental demand."""
    validate_required_columns(
        hourly_rentals,
        {"datetime_hour", "location_id", "total_rentals"},
        "hourly rentals",
    )
    validate_required_columns(
        prepared_weather,
        {"datetime_hour"},
        "prepared weather",
    )

    weather_features = prepared_weather.drop(
        columns=["id", "datetime"],
        errors="ignore",
    )

    enriched_rentals = hourly_rentals.merge(
        weather_features,
        on="datetime_hour",
        how="left",
    )

    return enriched_rentals.sort_values(
        ["datetime_hour", "location_id"],
    ).reset_index(drop=True)

def join_holiday_data(
    rentals_with_weather: pd.DataFrame,
    prepared_holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Join holiday information onto rental demand by date."""
    validate_required_columns(
        rentals_with_weather,
        {"datetime_hour", "location_id", "total_rentals"},
        "weather-enriched rentals",
    )
    validate_required_columns(
        prepared_holidays,
        {"date", "holiday"},
        "prepared holidays",
    )

    rentals = rentals_with_weather.copy()
    holidays = prepared_holidays.copy()

    rentals["date"] = rentals["datetime_hour"].dt.date

    enriched_rentals = rentals.merge(
        holidays.drop(columns=["id"], errors="ignore"),
        on="date",
        how="left",
    )

    enriched_rentals["is_holiday"] = enriched_rentals["holiday"].notna()

    return enriched_rentals.sort_values(
        ["datetime_hour", "location_id"],
    ).reset_index(drop=True)