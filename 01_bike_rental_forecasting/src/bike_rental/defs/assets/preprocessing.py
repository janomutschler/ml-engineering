"""Dagster assets for preprocessing bike rental datasets."""

import pandas as pd
from dagster import AssetExecutionContext, asset

from bike_rental.defs.constants import BIKE_RENTAL_FEATURE_COLUMNS
from bike_rental.defs.preprocessing.aggregation import aggregate_hourly_rental_activity
from bike_rental.defs.preprocessing.assembly import assemble_modeling_features
from bike_rental.defs.preprocessing.calendar_features import create_calendar_features
from bike_rental.defs.preprocessing.weather import clean_weather_data
from bike_rental.defs.utils.metadata import build_dataframe_metadata


@asset(io_manager_key="parquet_io_manager")
def hourly_rental_activity(
    context: AssetExecutionContext,
    booked_rentals: pd.DataFrame,
    direct_pickups: pd.DataFrame,
) -> pd.DataFrame:
    """Create hourly rental activity from rental events.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    booked_rentals : pd.DataFrame
        Booked rental records.
    direct_pickups : pd.DataFrame
        Direct pickup records.

    Returns
    -------
    pd.DataFrame
        Hourly rental activity dataset.

    """
    hourly_activity = aggregate_hourly_rental_activity(
        booked_rentals=booked_rentals,
        direct_pickups=direct_pickups,
    )

    context.log.info(
        "Created hourly rental activity with %s rows",
        len(hourly_activity),
    )

    context.add_output_metadata(
        build_dataframe_metadata(hourly_activity),
    )

    return hourly_activity


@asset(io_manager_key="parquet_io_manager")
def weather_cleaned(
    context: AssetExecutionContext,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Clean weather observations for downstream feature generation.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context used for logging and metadata.
    weather : pd.DataFrame
        Weather observations containing hourly weather measurements.

    Returns
    -------
    pd.DataFrame
        Cleaned weather observations with corrected anomalies and imputed
        humidity values.

    """
    cleaned_weather, cleaning_metadata = clean_weather_data(weather)

    context.add_output_metadata(
        build_dataframe_metadata(
            cleaned_weather,
            extra_metadata=cleaning_metadata,
        )
    )

    return cleaned_weather


@asset(io_manager_key="parquet_io_manager")
def calendar_features(
    context: AssetExecutionContext,
    hourly_rental_activity: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Create calendar features for the hourly rental timeline.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    hourly_rental_activity : pd.DataFrame
        Hourly rental activity dataset.
    holidays : pd.DataFrame
        Holiday calendar dataset.

    Returns
    -------
    pd.DataFrame
        Hourly calendar feature dataset.

    """
    features = create_calendar_features(
        hourly_rental_activity=hourly_rental_activity,
        holidays=holidays,
    )

    context.log.info("Created calendar features with %s rows", len(features))

    context.add_output_metadata(
        build_dataframe_metadata(
            features,
            extra_metadata={
                "weekend_hours": int(features["is_weekend"].sum()),
                "holiday_hours": int(features["is_holiday"].sum()),
            },
        )
    )

    return features


@asset(io_manager_key="parquet_io_manager")
def bike_rental_features(
    context: AssetExecutionContext,
    hourly_rental_activity: pd.DataFrame,
    weather_cleaned: pd.DataFrame,
    calendar_features: pd.DataFrame,
) -> pd.DataFrame:
    """Create the final bike rental feature dataset.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    hourly_rental_activity : pd.DataFrame
        Hourly rental activity dataset.
    weather_cleaned : pd.DataFrame
        Cleaned weather dataset.
    calendar_features : pd.DataFrame
        Calendar feature dataset.

    Returns
    -------
    pd.DataFrame
        Combined bike rental feature dataset.

    """
    feature_set = hourly_rental_activity.merge(
        weather_cleaned,
        left_on="datetime_hour",
        right_on="datetime",
        how="left",
    )

    feature_set = feature_set.merge(
        calendar_features,
        on="datetime_hour",
        how="left",
    )

    empty_hours_mask = feature_set["temperature_c"].isna()

    feature_set = feature_set.loc[~empty_hours_mask].reset_index(drop=True)

    removed_empty_hours = int(empty_hours_mask.sum())

    feature_set = feature_set[BIKE_RENTAL_FEATURE_COLUMNS]

    context.log.info(
        "Created bike rental feature dataset with %s rows",
        len(feature_set),
    )

    context.add_output_metadata(
        build_dataframe_metadata(
            feature_set,
            extra_metadata={
                "empty_hours_removed": removed_empty_hours,
            },
        )
    )

    return feature_set


@asset(io_manager_key="parquet_io_manager")
def modeling_feature_set(
    context: AssetExecutionContext,
    bike_rental_features: pd.DataFrame,
) -> pd.DataFrame:
    """Create the final modeling feature set used for bike rental forecasting.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    bike_rental_features : pd.DataFrame
        Prepared bike rental dataset produced by the preprocessing pipeline.

    Returns
    -------
    pd.DataFrame
        Modeling-ready dataset containing engineered forecasting features.

    """
    input_columns = bike_rental_features.shape[1]

    context.log.info(
        "Creating modeling feature set from %s input rows.",
        len(bike_rental_features),
    )

    df = assemble_modeling_features(bike_rental_features)

    rows_before_dropna = len(df)

    df = df.dropna().reset_index(drop=True)

    rows_removed = rows_before_dropna - len(df)

    context.log.info(
        "Removed %s rows without sufficient historical context.",
        rows_removed,
    )

    context.add_output_metadata(
        build_dataframe_metadata(
            df,
            extra_metadata={
                "rows_removed": rows_removed,
                "generated_features": df.shape[1] - input_columns,
            },
        )
    )

    return df
