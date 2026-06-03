"""Aggregation functions for the bike rental preprocessing pipeline."""

import pandas as pd


def aggregate_hourly_rental_activity(
    booked_rentals: pd.DataFrame,
    direct_pickups: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate rental events into a continuous hourly rental activity dataset.

    Parameters
    ----------
    booked_rentals : pd.DataFrame
        Booked rental records containing a ``datetime`` column.
    direct_pickups : pd.DataFrame
        Direct pickup records containing a ``datetime`` column.

    Returns
    -------
        Continuous hourly rental activity containing booked rentals, direct
        pickups, and total rentals. Hours without rental activity are retained
        and filled with zero counts.

    """
    booked = booked_rentals.assign(is_booked=1)[["datetime", "is_booked"]]
    direct = direct_pickups.assign(is_booked=0)[["datetime", "is_booked"]]

    rental_events = pd.concat([booked, direct], ignore_index=True)

    rental_events["datetime_hour"] = rental_events["datetime"].dt.floor("h")

    hourly_activity = (
        rental_events.groupby("datetime_hour")
        .agg(
            booked_rentals=("is_booked", "sum"),
            total_rentals=("is_booked", "size"),
        )
        .reset_index()
    )

    hourly_activity["direct_pickups"] = (
        hourly_activity["total_rentals"] - hourly_activity["booked_rentals"]
    )

    full_hour_grid = pd.DataFrame(
        {
            "datetime_hour": pd.date_range(
                start=hourly_activity["datetime_hour"].min(),
                end=hourly_activity["datetime_hour"].max(),
                freq="h",
            )
        }
    )

    hourly_activity = full_hour_grid.merge(
        hourly_activity,
        on="datetime_hour",
        how="left",
    )

    hourly_activity[["booked_rentals", "direct_pickups", "total_rentals"]] = (
        hourly_activity[["booked_rentals", "direct_pickups", "total_rentals"]].fillna(0).astype(int)
    )

    return hourly_activity.sort_values("datetime_hour", ignore_index=True)
