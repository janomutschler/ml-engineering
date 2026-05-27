"""Dagster assets for validating and preparing raw bike rental datasets."""

from dagster import MaterializeResult, asset

from bike_rental.defs.preprocessing.preparation import (
    prepare_holidays,
    prepare_operational_rentals,
    prepare_weather,
)
from bike_rental.defs.resources.paths import (
    HOLIDAYS_QUARANTINE_PATH,
    OPERATIONAL_RENTALS_QUARANTINE_PATH,
    QUARANTINE_DIR,
    WEATHER_QUARANTINE_PATH,
)
from bike_rental.defs.utils.metadata import build_dataframe_metadata, build_missing_weather_metadata


@asset(group_name="prepared_data")
def prepared_operational_rentals(raw_booked_rentals, raw_direct_pickups):
    """Materialize all valid structured operational rental records and quarantine invalid rows."""
    valid_rentals, invalid_rentals = prepare_operational_rentals(
        raw_booked_rentals,
        raw_direct_pickups,
    )

    if not invalid_rentals.empty:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        invalid_rentals.to_csv(OPERATIONAL_RENTALS_QUARANTINE_PATH, index=False)
    else:
        OPERATIONAL_RENTALS_QUARANTINE_PATH.unlink(missing_ok=True)

    return MaterializeResult(
        value=valid_rentals,
        metadata=build_dataframe_metadata(
            valid_rentals,
            extra_metadata={
                "quarantined_rows": len(invalid_rentals),
                "quarantine_path": str(OPERATIONAL_RENTALS_QUARANTINE_PATH),
            },
        ),
    )


@asset(group_name="prepared_data")
def prepared_weather(raw_weather):
    """Materialize validated structured weather records and quarantine invalid rows."""
    valid_weather, invalid_weather = prepare_weather(raw_weather)

    if not invalid_weather.empty:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        invalid_weather.to_csv(WEATHER_QUARANTINE_PATH, index=False)
    else:
        WEATHER_QUARANTINE_PATH.unlink(missing_ok=True)

    return MaterializeResult(
        value=valid_weather,
        metadata=build_dataframe_metadata(
            valid_weather,
            extra_metadata={
                "quarantined_rows": len(invalid_weather),
                "quarantine_path": str(WEATHER_QUARANTINE_PATH),
                "missing_weather_values": build_missing_weather_metadata(valid_weather),
            },
        ),
    )


@asset(group_name="prepared_data")
def prepared_holidays(raw_holidays):
    """Materialize validated structured holiday records and quarantine invalid rows."""
    valid_holidays, invalid_holidays = prepare_holidays(raw_holidays)

    if not invalid_holidays.empty:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        invalid_holidays.to_csv(HOLIDAYS_QUARANTINE_PATH, index=False)
    else:
        HOLIDAYS_QUARANTINE_PATH.unlink(missing_ok=True)

    return MaterializeResult(
        value=valid_holidays,
        metadata=build_dataframe_metadata(
            valid_holidays,
            extra_metadata={
                "quarantined_rows": len(invalid_holidays),
                "quarantine_path": str(HOLIDAYS_QUARANTINE_PATH),
            },
        ),
    )
