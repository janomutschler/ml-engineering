"""Utility functions for building metadata about dataframes in the bike rental pipeline."""

from dagster import MetadataValue

WEATHER_FEATURE_COLUMNS = [
    "conditions",
    "temperature_c",
    "perceived_temperature_c",
    "humidity",
    "windspeed_kmh",
]


def build_missing_weather_metadata(data) -> dict:
    """Build metadata about missing weather feature values."""
    return {column: int(data[column].isna().sum()) for column in WEATHER_FEATURE_COLUMNS}


def build_dataframe_metadata(
    data,
    preview_rows: int = 5,
    extra_metadata: dict | None = None,
) -> dict:
    """Build Dagster metadata for a dataframe."""
    metadata = {
        "rows": len(data),
        "column_types": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "preview": MetadataValue.md(data.head(preview_rows).to_markdown()),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata
