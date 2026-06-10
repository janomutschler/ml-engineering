"""Read published pipeline assets from LakeFS ``main`` for serving."""

import pandas as pd

from bike_rental.defs.resources.lakefs import LakeFSResource

# Must match the prefix used by LakeFSParquetIOManager.
_PREFIX = "processed"


def read_published_asset(lakefs: LakeFSResource, asset_name: str) -> pd.DataFrame:
    """Read a published Parquet asset from the ``main`` branch.

    Parameters
    ----------
    lakefs : LakeFSResource
        LakeFS connection.
    asset_name : str
        Asset name (without extension), e.g. ``bike_rental_features``.

    Returns
    -------
    pd.DataFrame
        The asset as last published to ``main``.

    """
    uri = lakefs.object_uri("main", f"{_PREFIX}/{asset_name}.parquet")
    return pd.read_parquet(uri, storage_options=lakefs.storage_options())
