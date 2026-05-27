from dagster import MetadataValue

def build_dataframe_metadata(
    data,
    preview_rows: int = 5,
    extra_metadata: dict | None = None,
) -> dict:
    """Build Dagster metadata for a dataframe."""
    metadata = {
        "rows": len(data),
        "column_types": {
            column: str(dtype)
            for column, dtype in data.dtypes.items()
        },
        "preview": MetadataValue.md(
            data.head(preview_rows).to_markdown()
        ),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata