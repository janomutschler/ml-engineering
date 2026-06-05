"""Shared constants used throughout the bike rental pipeline."""

BOOKED_RENTALS_FILE = "registered_bike_rentals.csv"
DIRECT_PICKUPS_FILE = "direct_pickup_bike_rentals.csv"
WEATHER_FILE = "weather.csv"
HOLIDAYS_FILE = "holidays.csv"

TRAIN_RATIO = 0.8

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

SELECTED_FEATURE_COLUMNS = [
    "temperature_c",
    "perceived_temperature_c",
    "humidity",
    "windspeed_kmh",
    "weekday",
    "is_weekend",
    "is_holiday",
    "conditions_clouds",
    "conditions_heavy_rain",
    "conditions_light_rain",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "lag_24h",
    "lag_168h",
    "same_weekday_hour_mean_4w",
    "same_hour_mean_7d",
]
