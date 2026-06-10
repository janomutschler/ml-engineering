"""Tests for the prediction API request validation."""

import pytest
from pydantic import ValidationError

from bike_rental.api.schemas import ForecastRequest, HourlyWeather


def _valid_hour(**overrides) -> dict:
    base = {
        "conditions": "clear",
        "temperature_c": 18.0,
        "perceived_temperature_c": 17.0,
        "humidity": 65.0,
        "windspeed_kmh": 6.0,
    }
    base.update(overrides)
    return base


def test_accepts_exactly_24_valid_hours():
    """It accepts a well-formed 24-hour request."""
    request = ForecastRequest(weather=[_valid_hour() for _ in range(24)])
    assert len(request.weather) == 24


def test_rejects_wrong_number_of_hours():
    """It rejects a request that is not exactly 24 hours."""
    with pytest.raises(ValidationError):
        ForecastRequest(weather=[_valid_hour() for _ in range(23)])


def test_rejects_unknown_condition():
    """It rejects a weather condition the model was not trained on."""
    with pytest.raises(ValidationError):
        HourlyWeather(**_valid_hour(conditions="snowing"))


def test_rejects_out_of_range_humidity():
    """It rejects humidity outside the 0-100 range."""
    with pytest.raises(ValidationError):
        HourlyWeather(**_valid_hour(humidity=150.0))


def test_rejects_negative_windspeed():
    """It rejects negative wind speed."""
    with pytest.raises(ValidationError):
        HourlyWeather(**_valid_hour(windspeed_kmh=-5.0))
