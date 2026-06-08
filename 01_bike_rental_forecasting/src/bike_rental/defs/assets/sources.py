"""Dagster assets for loading bike rental source datasets."""

import pandas as pd
from dagster import AssetExecutionContext, asset

from bike_rental.defs.constants import (
    BOOKED_RENTALS_FILE,
    DIRECT_PICKUPS_FILE,
    HOLIDAYS_FILE,
    WEATHER_FILE,
)
from bike_rental.defs.resources.data_loader import LocalDataLoader


@asset(io_manager_key="parquet_io_manager")
def booked_rentals(
    context: AssetExecutionContext,
    data_loader: LocalDataLoader,
) -> pd.DataFrame:
    """Load booked bike rental records.

    Returns
    -------
    pd.DataFrame
        booked rental records with timestamp columns parsed as datetime.

    """
    return data_loader.load_csv(
        context=context,
        file_name=BOOKED_RENTALS_FILE,
    )


@asset(io_manager_key="parquet_io_manager")
def direct_pickups(
    context: AssetExecutionContext,
    data_loader: LocalDataLoader,
) -> pd.DataFrame:
    """Load direct bike pickup records.

    Returns
    -------
    pd.DataFrame
        direct pickup records with timestamp columns parsed as datetime.

    """
    return data_loader.load_csv(
        context=context,
        file_name=DIRECT_PICKUPS_FILE,
    )


@asset(io_manager_key="parquet_io_manager")
def weather(
    context: AssetExecutionContext,
    data_loader: LocalDataLoader,
) -> pd.DataFrame:
    """Load weather observations.

    Returns
    -------
    pd.DataFrame
        weather observations with the hourly timestamp parsed as datetime.

    """
    return data_loader.load_csv(
        context=context,
        file_name=WEATHER_FILE,
    )


@asset(
    metadata={"datetime_columns": ["date"]},
    io_manager_key="parquet_io_manager",
)
def holidays(
    context: AssetExecutionContext,
    data_loader: LocalDataLoader,
) -> pd.DataFrame:
    """Load holiday calendar records.

    Returns
    -------
    pd.DataFrame
        holiday calendar records with the date column parsed as datetime.

    """
    return data_loader.load_csv(
        context=context,
        file_name=HOLIDAYS_FILE,
    )
