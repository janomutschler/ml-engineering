"""Utilities for building Dagster metadata about dataframes in the pipeline."""

import pandas as pd
from dagster import MetadataValue


def build_dataframe_metadata(
    data: pd.DataFrame | pd.Series,
    preview_rows: int = 5,
    extra_metadata: dict | None = None,
) -> dict:
    """Build standardized Dagster output metadata for a dataframe or series.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        The asset payload to summarize.
    preview_rows : int, default=5
        Number of leading rows to render as a Markdown preview.
    extra_metadata : dict | None, default=None
        Additional key/value metadata merged into the result.

    Returns
    -------
    dict
        Metadata with the row count, per-column dtypes, a Markdown preview, and
        any ``extra_metadata`` provided.

    """
    if isinstance(data, pd.DataFrame):
        column_types = {column: str(dtype) for column, dtype in data.dtypes.items()}
    else:
        column_types = {data.name or "value": str(data.dtype)}

    metadata = {
        "rows": len(data),
        "column_types": column_types,
        "preview": MetadataValue.md(data.head(preview_rows).to_markdown()),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata
