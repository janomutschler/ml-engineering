"""Dagster assets for preprocessing bike rental datasets."""

from dagster import MaterializeResult, asset

from bike_rental.defs.preprocessing.aggregation import aggregate_hourly_rentals
from bike_rental.defs.utils.metadata import build_dataframe_metadata


@asset(group_name="preprocessing")
def hourly_rentals(prepared_operational_rentals):
    """Materialize hourly location-level rental demand."""
    rentals = aggregate_hourly_rentals(prepared_operational_rentals)

    return MaterializeResult(
        value=rentals,
        metadata=build_dataframe_metadata(rentals),
    )