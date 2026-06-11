"""Build the model feature matrix for a day-ahead forecast request.

The lag and rolling features the model expects are backward-looking, so they
can be computed for the forecast horizon purely from published history. This
module reuses the *exact* feature assembly used in training
(:func:`assemble_modeling_features` and :func:`to_model_matrix`), which
guarantees the features served match the features trained on (no
training-serving skew).
"""

import numpy as np
import pandas as pd

from bike_rental.api.schemas import HourlyWeather
from bike_rental.defs.constants import TARGET_COLUMN
from bike_rental.defs.preprocessing.assembly import (
    assemble_modeling_features,
    to_model_matrix,
)
from bike_rental.defs.preprocessing.calendar_features import create_calendar_features

_WEATHER_FIELDS = (
    "conditions",
    "temperature_c",
    "perceived_temperature_c",
    "humidity",
    "windspeed_kmh",
)


def build_forecast_features(
    weather: list[HourlyWeather],
    history: pd.DataFrame,
    holidays: pd.DataFrame,
    feature_columns: list[str],
    horizon_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Build the feature matrix for the horizon following the last history point.

    Parameters
    ----------
    weather : list[HourlyWeather]
        Ordered hourly weather for the forecast horizon (entry ``i`` is the
        ``i``-th hour after the last observed data point).
    history : pd.DataFrame
        Published ``bike_rental_features`` (pre-modeling) with observed demand.
    holidays : pd.DataFrame
        Published holiday calendar (``date`` column).
    feature_columns : list[str]
        The exact model input columns (from the training configuration).
    horizon_hours : int, default=24
        Number of hours to forecast.

    Returns
    -------
    tuple[pd.DataFrame, pd.DatetimeIndex]
        The feature matrix (one row per horizon hour, columns == feature_columns)
        and the datetime index of the forecast hours.

    """
    history = history.sort_values("datetime_hour").reset_index(drop=True)
    start = history["datetime_hour"].max() + pd.Timedelta(hours=1)
    horizon_index = pd.date_range(start=start, periods=horizon_hours, freq="h")

    horizon = pd.DataFrame({"datetime_hour": horizon_index})
    calendar = create_calendar_features(horizon[["datetime_hour"]], holidays)
    horizon = horizon.merge(calendar, on="datetime_hour")

    for field in _WEATHER_FIELDS:
        horizon[field] = [getattr(hour, field) for hour in weather]

    horizon[TARGET_COLUMN] = np.nan

    combined = pd.concat([history, horizon], ignore_index=True)

    # Identical engineering sequence to the training pipeline. The horizon's
    # historical-demand features look backward into observed history, so the
    # unknown horizon target is never used as a model input.
    combined = assemble_modeling_features(combined)

    horizon_features = combined.tail(horizon_hours).reset_index(drop=True)

    # Project onto the exact training columns: condition categories absent from
    # this request become all-zero (matching the reference category at training
    # time) and the result is guaranteed purely numeric.
    return to_model_matrix(horizon_features, feature_columns), horizon_index
