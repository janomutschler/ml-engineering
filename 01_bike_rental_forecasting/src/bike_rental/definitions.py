"""Dagster asset and resource definitions for the bike rental project."""

from dagster import Definitions

from bike_rental.defs.assets.preparation import (
    prepared_holidays,
    prepared_operational_rentals,
    prepared_weather,
)
from bike_rental.defs.assets.preprocessing import (
    base_dataset,
    hourly_rentals,
    rentals_with_weather,
    rentals_with_weather_and_holidays,
)
from bike_rental.defs.assets.raw import (
    raw_booked_rentals,
    raw_direct_pickups,
    raw_holidays,
    raw_weather,
)
from bike_rental.defs.io_managers.csv_io_manager import CsvIOManager
from bike_rental.defs.resources.paths import PROCESSED_DATA_DIR

defs = Definitions(
    assets=[
        raw_booked_rentals,
        raw_direct_pickups,
        raw_weather,
        raw_holidays,
        prepared_operational_rentals,
        prepared_weather,
        prepared_holidays,
        hourly_rentals,
        rentals_with_weather,
        rentals_with_weather_and_holidays,
        base_dataset,
    ],
    resources={
        "csv_io_manager": CsvIOManager(
            base_path=str(PROCESSED_DATA_DIR),
        ),
    },
)
