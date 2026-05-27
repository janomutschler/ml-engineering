"""Preparation and validation utilities for raw bike rental datasets."""

import pandas as pd

from bike_rental.defs.preprocessing.quarantine import (
    build_quarantine_reason,
    split_valid_invalid_rows,
)
from bike_rental.defs.preprocessing.validate import validate_required_columns
from bike_rental.defs.resources.logging import logger

def prepare_operational_rentals(
    booked_rentals: pd.DataFrame,
    direct_pickups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Return validated operational rentals and quarantined invalid rows."""
	
	required_columns = {"id", "datetime", "location_id"}

	validate_required_columns(booked_rentals, required_columns, "booked rentals")
	validate_required_columns(direct_pickups, required_columns, "direct pickups")

	booked = booked_rentals[["id", "datetime", "location_id"]].copy()
	direct = direct_pickups[["id", "datetime", "location_id"]].copy()

	booked["source_dataset"] = "booked_rentals"
	direct["source_dataset"] = "direct_pickups"

	booked["is_booked"] = 1
	direct["is_booked"] = 0

	booked["is_duplicate_id"] = booked["id"].duplicated(keep=False)
	direct["is_duplicate_id"] = direct["id"].duplicated(keep=False)

	rentals = pd.concat([booked, direct], ignore_index=True)

	rentals["datetime"] = pd.to_datetime(rentals["datetime"], errors="coerce")
	rentals["datetime_hour"] = rentals["datetime"].dt.floor("h")

	reason_masks = {
		"duplicate id within source dataset": rentals["is_duplicate_id"],
		"missing or invalid datetime": rentals["datetime"].isna(),
		"missing or negative location_id": (
			rentals["location_id"].isna()
			| (rentals["location_id"] < 0)
		),
	}

	rentals["quarantine_reason"] = build_quarantine_reason(reason_masks)
	invalid_mask = rentals["quarantine_reason"] != ""

	valid_rentals, invalid_rentals = split_valid_invalid_rows(rentals, invalid_mask)

	valid_rentals = valid_rentals.drop(columns=["is_duplicate_id", "quarantine_reason"])
	valid_rentals = valid_rentals.sort_values(["datetime_hour", "location_id"])
	valid_rentals = valid_rentals.reset_index(drop=True)

	invalid_rentals = invalid_rentals.drop(columns=["is_duplicate_id"])
	invalid_rentals = invalid_rentals.reset_index(drop=True)

	logger.info(
		"Prepared operational rentals: %s valid rows, %s quarantined rows",
		len(valid_rentals),
		len(invalid_rentals),
	)

	return valid_rentals, invalid_rentals

def prepare_weather(weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return validated weather records and quarantined invalid rows."""
    required_columns = {"datetime"}

    validate_required_columns(weather, required_columns, "weather")

    prepared_weather = weather.copy()

    prepared_weather["datetime"] = pd.to_datetime(prepared_weather["datetime"], errors="coerce",)
    prepared_weather["datetime_hour"] = prepared_weather["datetime"].dt.floor("h")

    duplicate_hour_mask = prepared_weather["datetime_hour"].duplicated(keep=False)

    reason_masks = {
        "missing or invalid datetime": prepared_weather["datetime"].isna(),
        "duplicate datetime_hour": duplicate_hour_mask,
    }

    prepared_weather["quarantine_reason"] = build_quarantine_reason(reason_masks)
    invalid_mask = prepared_weather["quarantine_reason"] != ""

    valid_weather, invalid_weather = split_valid_invalid_rows(
        prepared_weather,
        invalid_mask,
    )

    valid_weather = valid_weather.drop(columns=["quarantine_reason"])
    valid_weather = valid_weather.reset_index(drop=True)

    invalid_weather = invalid_weather.reset_index(drop=True)

    logger.info(
        "Prepared weather: %s valid rows, %s quarantined rows",
        len(valid_weather),
        len(invalid_weather),
    )

    return valid_weather, invalid_weather

def prepare_holidays(
    holidays: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return validated holiday records and quarantined invalid rows."""
    required_columns = {"id", "date", "holiday"}

    validate_required_columns(holidays, required_columns, "holidays")

    prepared_holidays = holidays.copy()

    prepared_holidays["date"] = pd.to_datetime(prepared_holidays["date"], errors="coerce",).dt.date

    reason_masks = {
        "duplicate id": prepared_holidays["id"].duplicated(keep=False),
        "missing or invalid date": prepared_holidays["date"].isna(),
        "duplicate holiday date": prepared_holidays["date"].duplicated(keep=False),
        "missing holiday": prepared_holidays["holiday"].isna(),
    }

    prepared_holidays["quarantine_reason"] = build_quarantine_reason(reason_masks)

    invalid_mask = prepared_holidays["quarantine_reason"] != ""

    valid_holidays, invalid_holidays = split_valid_invalid_rows(
        prepared_holidays,
        invalid_mask,
    )

    valid_holidays = valid_holidays.drop(columns=["quarantine_reason"])
    valid_holidays = valid_holidays.reset_index(drop=True)

    invalid_holidays = invalid_holidays.reset_index(drop=True)

    logger.info(
        "Prepared holidays: %s valid rows, %s quarantined rows",
        len(valid_holidays),
        len(invalid_holidays),
    )

    return valid_holidays, invalid_holidays