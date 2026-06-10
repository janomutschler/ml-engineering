"""FastAPI application serving day-ahead bike rental demand forecasts.

The service loads the ``@champion`` model from the MLflow registry at startup,
computes features for the forecast horizon from data published to LakeFS
``main``, and returns hourly demand predictions. Deploying a new model is a
registry alias move plus a ``/reload`` — no code change or redeploy.
"""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from bike_rental.api.config import ApiSettings
from bike_rental.api.data import read_published_asset
from bike_rental.api.features import build_forecast_features
from bike_rental.api.model import load_champion
from bike_rental.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HourlyPrediction,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load settings and the champion model once at startup."""
    settings = ApiSettings.from_env()
    app.state.settings = settings
    app.state.champion = load_champion(
        settings.mlflow_tracking_uri,
        settings.registered_model_name,
    )
    yield


app = FastAPI(title="Bike Rental Demand Forecast API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report liveness and which model version and data commit are loaded."""
    champion = app.state.champion
    return HealthResponse(
        status="ok",
        model_name=app.state.settings.registered_model_name,
        model_version=champion.version,
        data_commit=champion.data_commit,
    )


@app.post("/reload", response_model=HealthResponse)
def reload_model() -> HealthResponse:
    """Reload the champion model from the registry.

    Lets the service pick up a newly promoted champion without a restart.
    """
    settings = app.state.settings
    app.state.champion = load_champion(
        settings.mlflow_tracking_uri,
        settings.registered_model_name,
    )
    return health()


@app.post("/predictions", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    """Forecast hourly demand for the 24 hours after the last published data."""
    settings = app.state.settings
    champion = app.state.champion
    lakefs = settings.lakefs()

    try:
        history = read_published_asset(lakefs, "bike_rental_features")
        holidays = read_published_asset(lakefs, "holidays")
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not read published data from LakeFS: {error}",
        ) from error

    feature_matrix, horizon_index = build_forecast_features(
        weather=request.weather,
        history=history,
        holidays=holidays,
        feature_columns=champion.feature_columns,
    )

    raw_predictions = champion.model.predict(feature_matrix)
    demand = np.clip(raw_predictions, 0, None).round().astype(int)

    predictions = [
        HourlyPrediction(
            datetime_hour=timestamp.isoformat(),
            hour=int(timestamp.hour),
            predicted_demand=int(value),
        )
        for timestamp, value in zip(horizon_index, demand, strict=True)
    ]

    return ForecastResponse(
        forecast_start=horizon_index[0].isoformat(),
        predictions=predictions,
        model_name=settings.registered_model_name,
        model_version=champion.version,
        data_commit=champion.data_commit,
    )
