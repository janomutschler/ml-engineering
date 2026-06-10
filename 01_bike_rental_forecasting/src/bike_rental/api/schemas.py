"""Request and response schemas for the prediction API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Weather conditions the model was trained on. "clear" is the reference
# category (encoded as all-zeros across the condition columns).
Conditions = Literal["clear", "clouds", "light_rain", "heavy_rain"]

FORECAST_HORIZON_HOURS = 24


class HourlyWeather(BaseModel):
    """Weather forecast for one hour of the forecast horizon."""

    conditions: Conditions
    temperature_c: float = Field(ge=-50, le=60)
    perceived_temperature_c: float = Field(ge=-50, le=60)
    humidity: float = Field(ge=0, le=100)
    windspeed_kmh: float = Field(ge=0, le=300)


class ForecastRequest(BaseModel):
    """A day-ahead forecast request.

    ``weather`` holds exactly 24 ordered hourly forecasts, one per hour of the
    24-hour horizon that immediately follows the last observed data point. The
    forecast date is not supplied by the caller; it is anchored to the end of
    the published history.
    """

    weather: list[HourlyWeather]

    @field_validator("weather")
    @classmethod
    def _exactly_one_day(cls, value: list[HourlyWeather]) -> list[HourlyWeather]:
        if len(value) != FORECAST_HORIZON_HOURS:
            raise ValueError(
                f"weather must contain exactly {FORECAST_HORIZON_HOURS} hourly entries, "
                f"got {len(value)}."
            )
        return value


class HourlyPrediction(BaseModel):
    """Predicted demand for a single hour."""

    datetime_hour: str
    hour: int
    predicted_demand: int


class ForecastResponse(BaseModel):
    """A full day-ahead demand forecast plus the lineage of the model used."""

    forecast_start: str
    predictions: list[HourlyPrediction]
    model_name: str
    model_version: str
    data_commit: str | None


class HealthResponse(BaseModel):
    """Liveness and loaded-model information."""

    status: str
    model_name: str
    model_version: str
    data_commit: str | None
