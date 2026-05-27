"""Dagster assets for preprocessing bike rental datasets."""

from dagster import MaterializeResult, asset

from bike_rental.defs.preprocessing.aggregation import aggregate_hourly_rentals
from bike_rental.defs.preprocessing.enrichment import join_holiday_data, join_weather_data
from bike_rental.defs.preprocessing.feature_engineering import add_time_features
from bike_rental.defs.utils.metadata import build_dataframe_metadata, build_missing_weather_metadata


@asset(group_name="preprocessing")
def hourly_rentals(prepared_operational_rentals):
    """Materialize hourly location-level rental demand."""
    rentals = aggregate_hourly_rentals(prepared_operational_rentals)

    return MaterializeResult(
        value=rentals,
        metadata=build_dataframe_metadata(rentals),
    )


@asset(group_name="preprocessing")
def rentals_with_weather(hourly_rentals, prepared_weather):
    """Materialize hourly rentals enriched with weather features."""
    rentals = join_weather_data(hourly_rentals, prepared_weather)

    return MaterializeResult(
        value=rentals,
        metadata=build_dataframe_metadata(
            rentals,
            extra_metadata={
                "missing_weather_values": build_missing_weather_metadata(rentals),
            },
        ),
    )


@asset(group_name="preprocessing")
def rentals_with_weather_and_holidays(rentals_with_weather, prepared_holidays):
    """Materialize rentals with weather enriched with holiday information."""
    rentals = join_holiday_data(rentals_with_weather, prepared_holidays)

    return MaterializeResult(
        value=rentals,
        metadata=build_dataframe_metadata(
            rentals,
            extra_metadata={
                "holiday_rows": int(rentals["is_holiday"].sum()),
                "missing_weather_values": build_missing_weather_metadata(rentals),
            },
        ),
    )


@asset(group_name="preprocessing")
def base_dataset(rentals_with_weather_and_holidays):
    """Materialize the curated base dataset for downstream analysis and ML workflows."""
    rentals = add_time_features(rentals_with_weather_and_holidays)

    return MaterializeResult(
        value=rentals,
        metadata=build_dataframe_metadata(
            rentals,
            extra_metadata={
                "missing_weather_values": build_missing_weather_metadata(rentals),
            },
        ),
    )
