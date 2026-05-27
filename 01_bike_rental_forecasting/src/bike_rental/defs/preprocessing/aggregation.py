import pandas as pd

from bike_rental.defs.preprocessing.validate import validate_required_columns

def aggregate_hourly_rentals(
	prepared_operational_rentals: pd.DataFrame,
) -> pd.DataFrame:
	"""Aggregate rental events to hourly location-level demand."""
	validate_required_columns(
		prepared_operational_rentals,
		{"id", "datetime_hour", "location_id", "is_booked"},
		"prepared operational rentals",
	)
	
	hourly_rentals = (
		prepared_operational_rentals
		.groupby(["datetime_hour", "location_id"], as_index=False)
		.agg(
			booked_rentals=("is_booked", "sum"),
			total_rentals=("id", "count"),
		)
	)

	hourly_rentals["direct_pickups"] = (
		hourly_rentals["total_rentals"] - hourly_rentals["booked_rentals"]
	)

	hourly_rentals = hourly_rentals[
		[
			"datetime_hour",
			"location_id",
			"booked_rentals",
			"direct_pickups",
			"total_rentals",
		]
	]
	return hourly_rentals.sort_values(
		["datetime_hour", "location_id"]
	).reset_index(drop=True)