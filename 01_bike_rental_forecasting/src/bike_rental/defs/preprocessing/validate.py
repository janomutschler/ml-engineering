"""Validation utilities for raw bike rental datasets."""

import pandas as pd

def validate_required_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Validate that all required columns exist."""
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"{dataset_name} is missing columns: {sorted(missing_columns)}")