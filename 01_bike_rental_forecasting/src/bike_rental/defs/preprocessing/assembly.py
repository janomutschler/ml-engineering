"""Single source of truth for turning pre-modeling features into a model matrix.

Both the training pipeline and the serving layer call these helpers, which is
what guarantees that the features a model is served match the features it was
trained on. Keeping the engineering sequence and the column projection in one
place removes the risk of the two paths drifting apart (training-serving skew).
"""

import pandas as pd

from bike_rental.defs.constants import TARGET_COLUMN
from bike_rental.defs.preprocessing.feature_transforms import (
    add_cyclical_features,
    add_historical_demand_features,
    one_hot_encode_column,
)


def assemble_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature-engineering sequence to pre-modeling features.

    This is the canonical definition of the feature pipeline: one-hot encoding
    of weather conditions, cyclical encodings of hour and month, and the
    backward-looking historical-demand features. The training asset and the
    prediction API both call it, so the engineered columns are identical on
    both paths by construction.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-modeling ``bike_rental_features`` rows (observed history and/or
        forecast-horizon rows). Must contain ``conditions``, ``hour``,
        ``month``, ``weekday`` and the target column.

    Returns
    -------
    pd.DataFrame
        ``df`` enriched with one-hot, cyclical, and historical-demand features.
        Rows are not dropped; callers decide how to handle rows that lack
        sufficient history for the lag features.

    """
    df = one_hot_encode_column(df, column="conditions")
    df = add_cyclical_features(df, column="hour", period=24)
    df = add_cyclical_features(df, column="month", period=12)
    df = add_historical_demand_features(
        df,
        target_column=TARGET_COLUMN,
        hour_column="hour",
        weekday_column="weekday",
    )
    return df


def to_model_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Project engineered features onto the exact model input columns.

    Reindexes ``df`` to ``feature_columns`` and casts to ``float64``. One-hot
    condition columns absent from this particular slice are filled with 0,
    which matches how the reference category (and any unseen condition) behaves
    at training time. Using this on both paths removes the former asymmetry,
    where serving tolerated missing columns via reindex but training selected
    columns directly and would raise ``KeyError``.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered features produced by :func:`assemble_modeling_features`.
    feature_columns : list[str]
        Exact model input columns, in the order the model expects them.

    Returns
    -------
    pd.DataFrame
        A purely numeric matrix with columns equal to ``feature_columns``.

    """
    return df.reindex(columns=feature_columns, fill_value=0).astype("float64")
