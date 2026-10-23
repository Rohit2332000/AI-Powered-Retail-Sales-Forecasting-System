import os
from pathlib import Path

import pandas as pd

from .databricks_client import get_connection


# ============================================================
# Databricks Gold Table
# ============================================================
CATALOG = "sales_ai"
SCHEMA = "gold"
TABLE = "forecast_results"


def load_forecast_results():
    """
    Load the Gold forecast-results dataset.

    Preferred source:
        - Databricks table configured by the active .env

    Optional fallback:
        - Local cache file defined by LOCAL_FORECAST_RESULTS_PATH

    Returns:
        pandas.DataFrame
    """

    local_dataset_path = os.getenv("LOCAL_FORECAST_RESULTS_PATH")

    if local_dataset_path:
        local_path = Path(local_dataset_path)

        if local_path.exists():
            suffix = local_path.suffix.lower()

            if suffix == ".csv":
                return pd.read_csv(local_path)

            if suffix in {".parquet", ".pq"}:
                return pd.read_parquet(local_path)

            if suffix in {".feather", ".ftr"}:
                return pd.read_feather(local_path)

            raise ValueError(
                "LOCAL_FORECAST_RESULTS_PATH must point to a .csv, .parquet, or .feather file."
            )

    query = f"""
        SELECT *
        FROM {CATALOG}.{SCHEMA}.{TABLE}
    """

    connection = None

    try:
        connection = get_connection()
        return pd.read_sql(query, connection)

    except Exception as exc:
        raise RuntimeError(
            "Failed to load forecast results from Databricks. "
            "Check that DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, and "
            "DATABRICKS_TOKEN in the .env file are valid and that the Databricks "
            "SQL warehouse/path is available. If you have a local cached file, set "
            "LOCAL_FORECAST_RESULTS_PATH to point to it."
        ) from exc

    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":

    df = load_forecast_results()

    print("=" * 60)
    print("FORECAST RESULTS LOADER TEST")
    print("=" * 60)

    print(f"Records : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample:")
    print(df.head())