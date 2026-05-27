"""Quarantine utilities for raw bike rental datasets."""

import pandas as pd

def build_quarantine_reason(reason_masks: dict[str, pd.Series]) -> pd.Series:
    """Build quarantine reason text from validation masks."""
    reasons = pd.Series("", index=next(iter(reason_masks.values())).index)

    for reason, mask in reason_masks.items():
        reasons = reasons.mask(
            mask,
            reasons.where(reasons == "", reasons + "; ") + reason,
        )

    return reasons


def split_valid_invalid_rows(
    data: pd.DataFrame,
    invalid_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into valid and invalid rows."""
    valid_rows = data.loc[~invalid_mask].copy()
    invalid_rows = data.loc[invalid_mask].copy()

    return valid_rows, invalid_rows