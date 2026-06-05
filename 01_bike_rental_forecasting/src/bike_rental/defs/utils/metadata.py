"""Utility functions for building metadata about dataframes in the bike rental pipeline."""

import pandas as pd
from dagster import MetadataValue, Output


def build_dataframe_metadata(
    data,
    preview_rows: int = 5,
    extra_metadata: dict | None = None,
) -> dict:
    """Build Dagster metadata for a dataframe."""
    if isinstance(data, pd.DataFrame):
        column_types = {column: str(dtype) for column, dtype in data.dtypes.items()}
    else:
        column_types = {data.name or "value": str(data.dtype)}
    metadata = {
        "rows": len(data),
        "columns_types": column_types,
        "preview": MetadataValue.md(data.head(preview_rows).to_markdown()),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata


def build_output(
    data: pd.DataFrame,
    output_name: str,
    extra_metadata: dict | None = None,
) -> Output:
    """Create a Dagster Output with standardized dataframe metadata.

    Parameters
    ----------
    data : pd.DataFrame
        Output dataframe.
    output_name : str
        Name of the Dagster output.
    extra_metadata : dict | None, default=None
        Additional metadata to attach.

    Returns
    -------
    Output
        Dagster output containing the dataframe and metadata.

    """
    return Output(
        data,
        output_name=output_name,
        metadata=build_dataframe_metadata(
            data,
            extra_metadata=extra_metadata,
        ),
    )
