"""Dagster assets for preprocessing bike rental datasets."""

from dagster import MaterializeResult, asset

from bike_rental.defs.preprocessing.aggregation import aggregate_hourly_rentals
from bike_rental.defs.utils.metadata import build_dataframe_metadata
from bike_rental.defs.preprocessing.enrichment import join_weather_data, join_holiday_data

@asset(group_name="preprocessing")
def hourly_rentals(prepared_operational_rentals):
	"""Materialize hourly location-level rental demand."""
	rentals = aggregate_hourly_rentals(prepared_operational_rentals)

	return MaterializeResult(
		value=rentals,
		metadata=build_dataframe_metadata(rentals),
	)

@asset(group_name="preprocessing")
def rentals_with_weather(hourly_rentals, prepared_weather):
	"""Materialize hourly rentals enriched with weather features."""
	rentals = join_weather_data(hourly_rentals, prepared_weather)

	weather_missing_values = {
		column: int(rentals[column].isna().sum())
		for column in [
			"conditions",
			"temperature_c",
			"perceived_temperature_c",
			"humidity",
			"windspeed_kmh",
		]
	}
	return MaterializeResult(
		value=rentals,
		metadata=build_dataframe_metadata(
			rentals,
			extra_metadata={
				"weather_missing_values": weather_missing_values,
			},
		),
	)

@asset(group_name="preprocessing")
def rentals_with_weather_and_holidays(rentals_with_weather, prepared_holidays):
    """Materialize rentals with weather enriched with holiday information."""
    rentals = join_holiday_data(rentals_with_weather, prepared_holidays)

    return MaterializeResult(
        value=rentals,
        metadata=build_dataframe_metadata(
            rentals,
            extra_metadata={
                "holiday_rows": int(rentals["is_holiday"].sum()),
            },
        ),
    )