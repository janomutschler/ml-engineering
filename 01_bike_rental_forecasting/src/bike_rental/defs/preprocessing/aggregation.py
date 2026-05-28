"""Aggregation functions for the bike rental preprocessing pipeline."""

import pandas as pd

from bike_rental.defs.preprocessing.validate import validate_required_columns

COUNT_COLUMNS = ["booked_rentals", "direct_pickups", "total_rentals"]
HOURLY_RENTALS_COLUMNS = ["datetime_hour", "location_id"] + COUNT_COLUMNS


def aggregate_hourly_rentals(
    prepared_operational_rentals: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate rental events into a complete hourly location-level demand table.

    The function aggregates operational rental events by hour and location and
    generates a complete grid of all hourly timestamps and locations. Missing
    rental combinations are filled with 0 to ensure that zero-demand periods are
    represented in the final dataset.
    """
    validate_required_columns(
        prepared_operational_rentals,
        {"id", "datetime_hour", "location_id", "is_booked"},
        "prepared operational rentals",
    )

    hourly_rentals = prepared_operational_rentals.groupby(
        ["datetime_hour", "location_id"], as_index=False
    ).agg(
        booked_rentals=("is_booked", "sum"),
        total_rentals=("id", "count"),
    )

    hourly_rentals["direct_pickups"] = (
        hourly_rentals["total_rentals"] - hourly_rentals["booked_rentals"]
    )

    hourly_rentals = hourly_rentals[HOURLY_RENTALS_COLUMNS]

    all_hours = pd.date_range(
        start=prepared_operational_rentals["datetime_hour"].min(),
        end=prepared_operational_rentals["datetime_hour"].max(),
        freq="h",
    )

    all_locations = prepared_operational_rentals["location_id"].unique()

    complete_grid = pd.MultiIndex.from_product(
        [all_hours, all_locations],
        names=["datetime_hour", "location_id"],
    ).to_frame(index=False)

    hourly_rentals = complete_grid.merge(
        hourly_rentals,
        on=["datetime_hour", "location_id"],
        how="left",
    )

    hourly_rentals[COUNT_COLUMNS] = hourly_rentals[COUNT_COLUMNS].fillna(0).astype(int)

    return hourly_rentals.sort_values(["datetime_hour", "location_id"]).reset_index(drop=True)
