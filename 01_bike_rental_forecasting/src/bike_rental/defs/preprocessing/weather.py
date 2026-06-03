"""Weather preprocessing transformations for bike rental forecasting."""

import pandas as pd


def clean_weather_data(
    weather: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean weather observations.

    Parameters
    ----------
    weather : pd.DataFrame
        Weather observations containing temperature, perceived temperature,
        humidity, and datetime columns.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Cleaned weather data and metadata describing the performed corrections.

    """
    cleaned_weather = weather.copy()
    cleaned_weather["datetime"] = cleaned_weather["datetime"].dt.floor("h")

    temperature_difference = (
        cleaned_weather["perceived_temperature_c"] - cleaned_weather["temperature_c"]
    ).abs()

    perceived_temperature_anomalies = temperature_difference > 20

    cleaned_weather.loc[
        perceived_temperature_anomalies,
        "perceived_temperature_c",
    ] = cleaned_weather.loc[
        perceived_temperature_anomalies,
        "temperature_c",
    ]

    humidity_zero_values = cleaned_weather["humidity"] == 0

    cleaned_weather.loc[humidity_zero_values, "humidity"] = pd.NA

    missing_humidity_count = int(cleaned_weather["humidity"].isna().sum())

    cleaned_weather = cleaned_weather.sort_values(
        "datetime",
        ignore_index=True,
    )

    cleaned_weather["humidity"] = (
        cleaned_weather["humidity"].interpolate(method="linear").ffill().bfill()
    )

    cleaning_metadata = {
        "perceived_temperature_values_corrected": int(perceived_temperature_anomalies.sum()),
        "humidity_zero_values_replaced": int(humidity_zero_values.sum()),
        "humidity_values_imputed": missing_humidity_count,
    }

    return cleaned_weather.drop(columns="id"), cleaning_metadata
