"""Dagster asset and resource definitions for the bike rental project."""

from dagster import Definitions

from bike_rental.defs.asset_checks.asset_checks import (
    booked_rentals_schema,
    direct_pickups_schema,
    holidays_schema,
    hourly_rental_activity_has_continuous_hours,
    weather_schema,
)
from bike_rental.defs.assets.preprocessing import (
    bike_rental_features,
    calendar_features,
    hourly_rental_activity,
    weather_cleaned,
)
from bike_rental.defs.assets.sources import (
    booked_rentals,
    direct_pickups,
    holidays,
    weather,
)
from bike_rental.defs.io_managers.csv_io_manager import LocalCsvIOManager
from bike_rental.defs.resources.data_loader import LocalDataLoader

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
    ],
    asset_checks=[
        booked_rentals_schema,
        direct_pickups_schema,
        weather_schema,
        holidays_schema,
        hourly_rental_activity_has_continuous_hours,
    ],
    resources={
        "csv_io_manager": LocalCsvIOManager(),
        "data_loader": LocalDataLoader(),
    },
)
