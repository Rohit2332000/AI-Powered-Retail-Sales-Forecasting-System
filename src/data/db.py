"""Databricks SQL query helpers for the forecasting API.

Reuses the existing PAT-based connection from src.data.databricks_client —
this module only adds parameterized query helpers on top of it.
"""

import logging
from contextlib import contextmanager
from typing import Any, Optional

from src.data.databricks_client import get_connection

logger = logging.getLogger("sales_forecasting_api.db")

FORECAST_TABLE = "sales_ai.gold.forecast_results"

FORECAST_COLUMNS = [
    "forecast_id",
    "forecast_date",
    "product_id",
    "category",
    "brand",
    "store_id",
    "store_name",
    "city",
    "forecast_horizon",
    "predicted_units",
    "prediction_timestamp",
    "model_name",
    "model_version",
]

# Named, parameterized queries for the /forecast?include_history=true merge --
# kept as top-level constants so it's easy to see, at a glance, exactly which
# table and columns feed actual_units vs predicted_units.

HISTORY_QUERY = """
    SELECT date, units_sold, product_id, store_id
    FROM sales_ai.gold.sales_features
    WHERE product_id = :product_id
      AND store_id = :store_id
      AND date >= date_sub(
          (SELECT MAX(date) FROM sales_ai.gold.sales_features
           WHERE product_id = :product_id AND store_id = :store_id),
          :history_days
      )
    ORDER BY date
"""

FORECAST_QUERY = """
    SELECT forecast_date, predicted_units, product_id, store_id
    FROM sales_ai.gold.forecast_results
    WHERE product_id = :product_id
      AND store_id = :store_id
    ORDER BY forecast_date
"""

# Named, parameterized queries for POST /forecast/explain's context builder --
# these are the ONLY facts handed to the LLM (see explain_context.py).

RECENT_FEATURES_QUERY = """
    SELECT date, units_sold, trend_7d, trend_28d, yoy_growth_pct, mom_growth_pct,
           rolling_mean_28, promotion_flag, days_since_last_promotion,
           holiday_flag, holiday_type, major_event_flag
    FROM sales_ai.gold.sales_features
    WHERE product_id = :product_id
      AND store_id = :store_id
      AND date >= date_sub(
          (SELECT MAX(date) FROM sales_ai.gold.sales_features
           WHERE product_id = :product_id AND store_id = :store_id),
          :days
      )
    ORDER BY date
"""

PRODUCT_STATIC_QUERY = """
    SELECT category, lifecycle_status, demand_class, price_elasticity_tier
    FROM sales_ai.silver.products
    WHERE product_id = :product_id
"""

# Named, parameterless queries backing the frontend's product/store selectors.

PRODUCTS_LIST_QUERY = """
    SELECT product_id, product_name, category
    FROM sales_ai.silver.products
    ORDER BY product_id
"""

STORES_LIST_QUERY = """
    SELECT store_id, store_name, city, region
    FROM sales_ai.silver.stores
    ORDER BY store_id
"""

# Named, parameterized queries backing analytics.py's business-question
# functions (used by both the /chat tool-calling path and the /chat/quick
# deterministic path, plus the /kpis dashboard summary). Each answers one
# real question with an aggregate SQL query -- no hardcoded numbers.

TOP_CATEGORY_QUERY = """
    SELECT category, SUM(units_sold) AS total_units
    FROM sales_ai.gold.sales_features
    WHERE date >= date_sub((SELECT MAX(date) FROM sales_ai.gold.sales_features), :days)
    GROUP BY category
    ORDER BY total_units DESC
"""

TOP_CATEGORY_BY_STORE_QUERY = """
    SELECT category, SUM(units_sold) AS total_units
    FROM sales_ai.gold.sales_features
    WHERE store_id = :store_id
      AND date >= date_sub((SELECT MAX(date) FROM sales_ai.gold.sales_features), :days)
    GROUP BY category
    ORDER BY total_units DESC
"""

TOP_PRODUCT_QUERY = """
    SELECT sf.product_id, p.product_name, SUM(sf.units_sold) AS total_units
    FROM sales_ai.gold.sales_features sf
    JOIN sales_ai.silver.products p ON p.product_id = sf.product_id
    WHERE sf.date >= date_sub((SELECT MAX(date) FROM sales_ai.gold.sales_features), :days)
    GROUP BY sf.product_id, p.product_name
    ORDER BY total_units DESC
"""

TOP_PRODUCT_BY_STORE_QUERY = """
    SELECT sf.product_id, p.product_name, SUM(sf.units_sold) AS total_units
    FROM sales_ai.gold.sales_features sf
    JOIN sales_ai.silver.products p ON p.product_id = sf.product_id
    WHERE sf.store_id = :store_id
      AND sf.date >= date_sub((SELECT MAX(date) FROM sales_ai.gold.sales_features), :days)
    GROUP BY sf.product_id, p.product_name
    ORDER BY total_units DESC
"""

CATEGORY_COMPARISON_QUERY = """
    SELECT category, SUM(units_sold) AS total_units, AVG(units_sold) AS avg_daily_units
    FROM sales_ai.gold.sales_features
    WHERE category IN (:category_a, :category_b)
      AND date >= date_sub((SELECT MAX(date) FROM sales_ai.gold.sales_features), :days)
    GROUP BY category
"""

STORE_PERFORMANCE_QUERY = """
    SELECT
        s.store_name,
        AVG(CASE WHEN sf.date >= date_sub(anchor.max_date, :recent_days) THEN sf.units_sold END) AS recent_avg,
        AVG(CASE
            WHEN sf.date >= date_sub(anchor.max_date, :baseline_days)
             AND sf.date < date_sub(anchor.max_date, :recent_days)
            THEN sf.units_sold
        END) AS baseline_avg
    FROM sales_ai.gold.sales_features sf
    JOIN sales_ai.silver.stores s ON s.store_id = sf.store_id
    CROSS JOIN (SELECT MAX(date) AS max_date FROM sales_ai.gold.sales_features) anchor
    WHERE sf.store_id = :store_id
    GROUP BY s.store_name
"""

TRENDING_DOWN_PRODUCTS_QUERY = """
    WITH latest AS (
        SELECT product_id, store_id, date, mom_growth_pct,
               ROW_NUMBER() OVER (PARTITION BY product_id, store_id ORDER BY date DESC) AS rn
        FROM sales_ai.gold.sales_features
    )
    SELECT l.product_id, p.product_name, AVG(l.mom_growth_pct) AS avg_mom_growth_pct
    FROM latest l
    JOIN sales_ai.silver.products p ON p.product_id = l.product_id
    WHERE l.rn = 1
    GROUP BY l.product_id, p.product_name
    HAVING AVG(l.mom_growth_pct) <= :threshold_pct
    ORDER BY avg_mom_growth_pct ASC
"""

TRENDING_DOWN_STORES_QUERY = """
    WITH latest AS (
        SELECT store_id, product_id, date, mom_growth_pct,
               ROW_NUMBER() OVER (PARTITION BY product_id, store_id ORDER BY date DESC) AS rn
        FROM sales_ai.gold.sales_features
    )
    SELECT l.store_id, s.store_name, AVG(l.mom_growth_pct) AS avg_mom_growth_pct
    FROM latest l
    JOIN sales_ai.silver.stores s ON s.store_id = l.store_id
    WHERE l.rn = 1
    GROUP BY l.store_id, s.store_name
    HAVING AVG(l.mom_growth_pct) <= :threshold_pct
    ORDER BY avg_mom_growth_pct ASC
"""

PREDICTED_REVENUE_QUERY = """
    SELECT SUM(fr.predicted_units * p.base_price) AS total_predicted_revenue,
           SUM(fr.predicted_units) AS total_predicted_units
    FROM sales_ai.gold.forecast_results fr
    JOIN sales_ai.silver.products p ON p.product_id = fr.product_id
"""


def run_query(query: str, params: Optional[dict] = None, connection=None) -> list[dict[str, Any]]:
    """Run a parameterized SQL query and return rows as a list of dicts.

    Opening a Databricks connection costs several seconds of its own (session
    handshake), separate from the query itself. By default this function
    pays that cost every call -- fine for a single query, but when several
    queries run back to back (e.g. /kpis' summary) it adds up fast enough to
    blow past a client-side timeout. Pass an existing `connection` (see
    connection_scope()) to reuse it and skip the per-call handshake.
    """

    logger.info("Executing query: %s | params=%s", " ".join(query.split()), params or {})

    owns_connection = connection is None
    if owns_connection:
        connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params or {})
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    finally:
        if owns_connection:
            connection.close()


@contextmanager
def connection_scope():
    """Open one Databricks connection for several queries run back to back,
    so only one session-open cost is paid instead of one per query. Pass the
    yielded connection to run_query(..., connection=...) for each call."""

    connection = get_connection()
    try:
        yield connection
    finally:
        connection.close()


def fetch_forecasts(
    product_id: Optional[str] = None,
    store_id: Optional[str] = None,
    category: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Fetch forecast rows, optionally filtered by product_id, store_id, category.

    Filter values are always passed as bound query parameters (never
    string-formatted into the SQL) to prevent injection.
    """

    filters = []
    params: dict[str, Any] = {}

    if product_id is not None:
        filters.append("product_id = :product_id")
        params["product_id"] = product_id

    if store_id is not None:
        filters.append("store_id = :store_id")
        params["store_id"] = store_id

    if category is not None:
        filters.append("category = :category")
        params["category"] = category

    query = f"SELECT {', '.join(FORECAST_COLUMNS)} FROM {FORECAST_TABLE}"

    if filters:
        query += " WHERE " + " AND ".join(filters)

    return run_query(query, params)


def check_connection() -> None:
    """Run a trivial query to verify the Databricks connection is alive."""

    run_query("SELECT 1")


def fetch_distinct_products() -> list[dict[str, Any]]:
    """Fetch the distinct product_id/category pairs present in forecast_results."""

    query = f"SELECT DISTINCT product_id, category FROM {FORECAST_TABLE} ORDER BY product_id"
    return run_query(query)


def fetch_history_actuals(product_id: str, store_id: str, history_days: int) -> list[dict[str, Any]]:
    """Real observed sales for one series, from sales_ai.gold.sales_features --
    the last `history_days` days relative to that series' own most recent date."""

    return run_query(
        HISTORY_QUERY,
        {"product_id": product_id, "store_id": store_id, "history_days": history_days},
    )


def fetch_forecast_predictions(product_id: str, store_id: str) -> list[dict[str, Any]]:
    """Model output for one series, from sales_ai.gold.forecast_results."""

    return run_query(FORECAST_QUERY, {"product_id": product_id, "store_id": store_id})


def fetch_recent_sales_features(product_id: str, store_id: str, days: int = 90) -> list[dict[str, Any]]:
    """Recent actuals + engineered features for one series, from
    sales_ai.gold.sales_features -- feeds the /forecast/explain context builder."""

    return run_query(
        RECENT_FEATURES_QUERY,
        {"product_id": product_id, "store_id": store_id, "days": days},
    )


def fetch_product_static(product_id: str) -> Optional[dict[str, Any]]:
    """Static product context (category, lifecycle_status, demand_class,
    price_elasticity_tier) from sales_ai.silver.products, or None if unknown."""

    rows = run_query(PRODUCT_STATIC_QUERY, {"product_id": product_id})
    return rows[0] if rows else None


def fetch_products() -> list[dict[str, Any]]:
    """All products (product_id, product_name, category) for the frontend selector."""

    return run_query(PRODUCTS_LIST_QUERY)


def fetch_stores() -> list[dict[str, Any]]:
    """All stores (store_id, store_name, city, region) for the frontend selector."""

    return run_query(STORES_LIST_QUERY)


def fetch_category_totals(
    days: int = 90, store_id: Optional[str] = None, connection=None
) -> list[dict[str, Any]]:
    """Total actual units sold per category over the last `days` days, best first.
    Scoped to one store if store_id is given, otherwise across all stores."""

    if store_id is not None:
        return run_query(TOP_CATEGORY_BY_STORE_QUERY, {"days": days, "store_id": store_id}, connection=connection)
    return run_query(TOP_CATEGORY_QUERY, {"days": days}, connection=connection)


def fetch_product_totals(
    days: int = 90, store_id: Optional[str] = None, connection=None
) -> list[dict[str, Any]]:
    """Total actual units sold per product over the last `days` days, best first.
    Scoped to one store if store_id is given, otherwise across all stores."""

    if store_id is not None:
        return run_query(TOP_PRODUCT_BY_STORE_QUERY, {"days": days, "store_id": store_id}, connection=connection)
    return run_query(TOP_PRODUCT_QUERY, {"days": days}, connection=connection)


def fetch_category_comparison(category_a: str, category_b: str, days: int = 90) -> list[dict[str, Any]]:
    """Total/average actual units sold for two categories over the last `days` days."""

    return run_query(
        CATEGORY_COMPARISON_QUERY,
        {"category_a": category_a, "category_b": category_b, "days": days},
    )


def fetch_store_performance(store_id: str, recent_days: int = 30, baseline_days: int = 180) -> Optional[dict[str, Any]]:
    """One store's recent average daily units sold vs. its own longer-run baseline average."""

    rows = run_query(
        STORE_PERFORMANCE_QUERY,
        {"store_id": store_id, "recent_days": recent_days, "baseline_days": baseline_days},
    )
    return rows[0] if rows else None


def fetch_trending_down(level: str, threshold_pct: float = -10.0, connection=None) -> list[dict[str, Any]]:
    """Products or stores whose latest month-over-month growth is at/below threshold_pct.

    level must be "product" or "store".
    """

    query = TRENDING_DOWN_PRODUCTS_QUERY if level == "product" else TRENDING_DOWN_STORES_QUERY
    return run_query(query, {"threshold_pct": threshold_pct}, connection=connection)


def fetch_predicted_revenue(connection=None) -> Optional[dict[str, Any]]:
    """Total predicted units/revenue across the whole forecast, using each
    product's base_price as a revenue proxy (forecast_results has no price
    of its own -- the actual future price is not yet known)."""

    rows = run_query(PREDICTED_REVENUE_QUERY, connection=connection)
    return rows[0] if rows else None
