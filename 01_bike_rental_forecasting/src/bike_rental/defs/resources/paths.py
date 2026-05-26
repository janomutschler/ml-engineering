from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
QUARANTINE_DIR = DATA_DIR / "quarantine"

BOOKED_RENTALS_PATH = RAW_DATA_DIR / "registered_bike_rentals.csv"
DIRECT_PICKUPS_PATH = RAW_DATA_DIR / "direct_pickup_bike_rentals.csv"
WEATHER_PATH = RAW_DATA_DIR / "weather.csv"
HOLIDAYS_PATH = RAW_DATA_DIR / "holidays.csv"

FINAL_DATASET_PATH = PROCESSED_DATA_DIR / "final_preprocessed_dataset.csv"