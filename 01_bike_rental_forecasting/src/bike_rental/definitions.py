from dagster import Definitions

from bike_rental.defs.assets.raw import raw_booked_rentals, raw_direct_pickups, raw_holidays, raw_weather
from bike_rental.defs.assets.preparation import prepared_operational_rentals, prepared_weather, prepared_holidays

defs = Definitions(
    assets=[
        raw_booked_rentals,
        raw_direct_pickups,
        raw_weather,
        raw_holidays,
        prepared_operational_rentals,
		prepared_weather,
		prepared_holidays,
    ]
)