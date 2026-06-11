"""Send an example day-ahead forecast request and print the demand curve.

Usage (with the API running via `make api`):

    uv run python scripts/api_example_request.py

Doubles as a smoke check: a non-zero, daily-shaped curve means the model,
feature pipeline, and data reads are all working end to end.
"""

import os

import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8001")

# A plausible day: cool clear night, light rain through the morning, warm clear
# afternoon, cooling off in the evening.
_DAY = [
    ("clear", 12),
    ("clear", 11),
    ("clear", 11),
    ("clear", 10),
    ("clouds", 10),
    ("clouds", 11),
    ("light_rain", 12),
    ("light_rain", 13),
    ("light_rain", 14),
    ("clouds", 16),
    ("clouds", 18),
    ("clear", 20),
    ("clear", 22),
    ("clear", 23),
    ("clear", 23),
    ("clear", 22),
    ("clouds", 21),
    ("clouds", 19),
    ("clear", 17),
    ("clear", 16),
    ("clear", 15),
    ("clear", 14),
    ("clear", 13),
    ("clear", 13),
]


def _weather_payload() -> list[dict]:
    return [
        {
            "conditions": conditions,
            "temperature_c": float(temp),
            "perceived_temperature_c": float(temp - 1),
            "humidity": 70.0,
            "windspeed_kmh": 6.0,
        }
        for conditions, temp in _DAY
    ]


def main() -> None:
    """Call the prediction endpoint and print the forecast as a simple bar chart."""
    health = requests.get(f"{API_URL}/health", timeout=10).json()
    print(f"Model: {health['model_name']} v{health['model_version']}")
    print(f"Data commit: {health['data_commit']}\n")

    response = requests.post(
        f"{API_URL}/predictions",
        json={"weather": _weather_payload()},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()

    print(f"Forecast starting {result['forecast_start']}:\n")
    peak = max(p["predicted_demand"] for p in result["predictions"]) or 1
    for prediction in result["predictions"]:
        demand = prediction["predicted_demand"]
        bar = "█" * round(40 * demand / peak)
        print(f"  {prediction['hour']:02d}:00  {demand:4d}  {bar}")


if __name__ == "__main__":
    main()
