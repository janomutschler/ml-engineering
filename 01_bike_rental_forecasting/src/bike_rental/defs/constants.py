"""Shared constants used throughout the bike rental pipeline."""

BOOKED_RENTALS_FILE = "registered_bike_rentals.csv"
DIRECT_PICKUPS_FILE = "direct_pickup_bike_rentals.csv"
WEATHER_FILE = "weather.csv"
HOLIDAYS_FILE = "holidays.csv"

TARGET_COLUMN = "total_rentals"

BOOKED_RENTALS_COLUMNS = {
    "id",
    "user_id",
    "location_id",
    "datetime",
}

DIRECT_PICKUPS_COLUMNS = {
    "id",
    "user_id",
    "location_id",
    "datetime",
}

WEATHER_COLUMNS = {
    "datetime",
    "conditions",
    "temperature_c",
    "perceived_temperature_c",
    "humidity",
    "windspeed_kmh",
}

HOLIDAYS_COLUMNS = {
    "date",
    "holiday",
}

BIKE_RENTAL_FEATURE_COLUMNS = [
    "datetime_hour",
    "conditions",
    "temperature_c",
    "perceived_temperature_c",
    "humidity",
    "windspeed_kmh",
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "is_holiday",
    "booked_rentals",
    "direct_pickups",
    "total_rentals",
]
