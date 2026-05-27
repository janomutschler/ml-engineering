"""Dagster assets for loading raw bike rental source datasets."""

import pandas as pd
from dagster import MaterializeResult, asset

from bike_rental.defs.resources.logging import logger
from bike_rental.defs.resources.paths import (
    BOOKED_RENTALS_PATH,
    DIRECT_PICKUPS_PATH,
    HOLIDAYS_PATH,
    WEATHER_PATH,
)
from bike_rental.defs.utils.metadata import build_dataframe_metadata


def _load_csv(path):
    """Load a CSV file from disk.

    path : pathlib.Path
        Path to the CSV file.

    Returns: pandas.DataFrame
        Loaded CSV data.
    """
    logger.info("Loading CSV file from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %s rows and %s columns from %s", len(df), len(df.columns), path)
    return df


@asset(group_name="raw_data")
def raw_booked_rentals():
    """Load raw booked bike rental records."""
    df = _load_csv(BOOKED_RENTALS_PATH)

    return MaterializeResult(
        value=df,
        metadata=build_dataframe_metadata(df),
    )


@asset(group_name="raw_data")
def raw_direct_pickups():
    """Load raw direct bike pickup records."""
    df = _load_csv(DIRECT_PICKUPS_PATH)

    return MaterializeResult(
        value=df,
        metadata=build_dataframe_metadata(df),
    )


@asset(group_name="raw_data")
def raw_weather():
    """Load raw weather records."""
    df = _load_csv(WEATHER_PATH)

    return MaterializeResult(
        value=df,
        metadata=build_dataframe_metadata(df),
    )


@asset(group_name="raw_data")
def raw_holidays():
    """Load raw holiday calendar records."""
    df = _load_csv(HOLIDAYS_PATH)

    return MaterializeResult(
        value=df,
        metadata=build_dataframe_metadata(df),
    )
