"""Enrichment functions for the bike rental preprocessing pipeline."""

import pandas as pd

from bike_rental.defs.preprocessing.validate import validate_required_columns


def join_weather_data(
    hourly_rentals: pd.DataFrame,
    prepared_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Join hourly weather features onto hourly rental demand.

    Rows are first enriched with weather observations based on the hourly timestamp.

    The preprocessing pipeline generates a complete hourly location grid to ensure
    that zero-demand periods are represented in the forecasting dataset. However,
    some timestamps do not exist in the weather dataset and may also represent
    periods with incomplete operational source coverage rather than true zero-demand
    hours.

    To avoid introducing misleading observations into the machine learning dataset,
    timestamps without both weather coverage and observed rental activity are
    removed after the join step. Any remaining missing weather
    values are treated as a data quality issue and raise an error instead of
    being silently imputed.
    """
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

    weather_columns = [column for column in weather_features.columns if column != "datetime_hour"]

    enriched_rentals["has_weather_data"] = ~enriched_rentals[weather_columns].isna().all(axis=1)

    timestamp_activity = enriched_rentals.groupby("datetime_hour")["total_rentals"].transform("sum")

    valid_timestamp = enriched_rentals["has_weather_data"] | (timestamp_activity > 0)

    enriched_rentals = enriched_rentals.loc[valid_timestamp].copy()

    missing_weather_counts = enriched_rentals[weather_columns].isna().sum()

    if missing_weather_counts.any():
        raise ValueError(
            "Weather join produced partially missing weather features after filtering. "
            f"Missing counts: {missing_weather_counts[missing_weather_counts > 0].to_dict()}"
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
