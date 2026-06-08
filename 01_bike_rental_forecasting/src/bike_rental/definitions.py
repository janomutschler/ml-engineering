"""Dagster asset and resource definitions for the bike rental project."""

from dagster import Definitions

from bike_rental.defs.asset_checks.asset_checks import (
    booked_rentals_schema,
    direct_pickups_schema,
    holidays_schema,
    hourly_rental_activity_has_continuous_hours,
    weather_schema,
)
from bike_rental.defs.assets.models import trained_forecasting_model
from bike_rental.defs.assets.preprocessing import (
    bike_rental_features,
    calendar_features,
    hourly_rental_activity,
    modeling_feature_set,
    weather_cleaned,
)
from bike_rental.defs.assets.sources import (
    booked_rentals,
    direct_pickups,
    holidays,
    weather,
)
from bike_rental.defs.constants import SELECTED_FEATURE_COLUMNS, TARGET_COLUMN
from bike_rental.defs.io_managers.io_manager import LocalParquetIOManager
from bike_rental.defs.resources.data_loader import LocalDataLoader
from bike_rental.defs.resources.training_config import TrainingConfigResource

defs = Definitions(
    assets=[
        booked_rentals,
        direct_pickups,
        weather,
        holidays,
        hourly_rental_activity,
        weather_cleaned,
        calendar_features,
        bike_rental_features,
        modeling_feature_set,
        trained_forecasting_model,
    ],
    asset_checks=[
        booked_rentals_schema,
        direct_pickups_schema,
        weather_schema,
        holidays_schema,
        hourly_rental_activity_has_continuous_hours,
    ],
    resources={
        "parquet_io_manager": LocalParquetIOManager(),
        "data_loader": LocalDataLoader(),
        "training_config": TrainingConfigResource(
            feature_columns=SELECTED_FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            model_type="xgboost",
        ),
    },
)
