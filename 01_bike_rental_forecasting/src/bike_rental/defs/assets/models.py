"""Dagster assets for training and persisting forecasting models."""

from pathlib import Path

import joblib
import pandas as pd
from dagster import AssetExecutionContext, asset
from xgboost import XGBRegressor

from bike_rental.defs.constants import TARGET_COLUMN

MODEL_PATH = "data/processed/models/xgboost_forecasting_model.joblib"


@asset
def xgboost_forecasting_model(
    context: AssetExecutionContext,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> str:
    """Train and persist the selected XGBoost forecasting model.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    X_train : pd.DataFrame
        Training feature dataset.
    y_train : pd.DataFrame
        Training target dataset.

    Returns
    -------
    str
        Path to the persisted trained model artifact.

    """
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )

    y_train_series = y_train[TARGET_COLUMN]

    model.fit(X_train, y_train_series)

    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    context.add_output_metadata(
        {
            "model_path": str(model_path),
            "model_type": "XGBRegressor",
            "training_rows": len(X_train),
            "feature_columns": len(X_train.columns),
            "target_column": TARGET_COLUMN,
        }
    )

    return str(model_path)
