"""Tests for weather preprocessing transformations."""

import pandas as pd

from bike_rental.defs.preprocessing.weather import clean_weather_data


def test_clean_weather_data_corrects_temperature_and_humidity_anomalies():
    """It fixes suspicious perceived temperature and humidity values."""
    weather = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 11:00:00",
                    "2024-01-01 12:00:00",
                ]
            ),
            "conditions": ["Cloudy", "Light Rain", "Cloudy"],
            "temperature_c": [10.0, 12.0, 14.0],
            "perceived_temperature_c": [10.0, 40.0, 14.0],
            "humidity": [80.0, 0.0, 90.0],
            "windspeed_kmh": [5.0, 6.0, 7.0],
        }
    )

    cleaned_weather, metadata = clean_weather_data(weather)

    assert cleaned_weather.loc[1, "perceived_temperature_c"] == 12.0
    assert cleaned_weather.loc[1, "humidity"] == 85.0

    assert metadata["perceived_temperature_values_corrected"] == 1
    assert metadata["humidity_zero_values_replaced"] == 1
    assert metadata["humidity_values_imputed"] == 1
