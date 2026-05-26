from dagster import Definitions

from bike_rental.defs.assets.raw import raw_booked_rentals, raw_direct_pickups, raw_holidays, raw_weather

defs = Definitions(
    assets=[
        raw_booked_rentals,
        raw_direct_pickups,
        raw_weather,
        raw_holidays,
    ]
)