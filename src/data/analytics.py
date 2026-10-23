"""Business-question analytics functions, each backed by a real SQL query.

Mirrors explain_context.py's role for /forecast/explain: this module owns
computation only (rounding, percentage math, thresholding) on top of rows
already fetched via db.py -- no SQL lives here, and no LLM call happens
here either. These functions are the single source of truth used by both
the /chat tool-calling path, the deterministic /chat/quick path, and the
/kpis dashboard summary, so all three always agree with each other.
"""

from typing import Any, Optional

from src.data import db

DEFAULT_TREND_THRESHOLD_PCT = -10.0


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / old * 100, 2)


def get_top_category(days: int = 90, store_id: Optional[str] = None, connection=None) -> dict[str, Any]:
    """The best-selling product category by total actual units sold.
    Scoped to one store if store_id is given, otherwise across all stores."""

    rows = db.fetch_category_totals(days=days, store_id=store_id, connection=connection)
    if not rows:
        return {"days": days, "store_id": store_id, "category": None, "total_units": None}

    top = rows[0]
    return {
        "days": days,
        "store_id": store_id,
        "category": top["category"],
        "total_units": round(top["total_units"], 2),
    }


def get_top_product(days: int = 90, store_id: Optional[str] = None, connection=None) -> dict[str, Any]:
    """The best-selling product by total actual units sold.
    Scoped to one store if store_id is given, otherwise across all stores."""

    rows = db.fetch_product_totals(days=days, store_id=store_id, connection=connection)
    if not rows:
        return {"days": days, "store_id": store_id, "product_id": None, "product_name": None, "total_units": None}

    top = rows[0]
    return {
        "days": days,
        "store_id": store_id,
        "product_id": top["product_id"],
        "product_name": top["product_name"],
        "total_units": round(top["total_units"], 2),
    }


def compare_categories(category_a: str, category_b: str, days: int = 90) -> dict[str, Any]:
    """Total/average actual units sold for two categories, head to head."""

    rows = db.fetch_category_comparison(category_a, category_b, days=days)
    by_category = {row["category"]: row for row in rows}

    def _entry(name: str) -> Optional[dict[str, Any]]:
        row = by_category.get(name)
        if row is None:
            return None
        return {"total_units": round(row["total_units"], 2), "avg_daily_units": round(row["avg_daily_units"], 2)}

    entry_a, entry_b = _entry(category_a), _entry(category_b)
    leader = None
    if entry_a and entry_b:
        leader = category_a if entry_a["total_units"] >= entry_b["total_units"] else category_b

    return {
        "days": days,
        "category_a": {"name": category_a, **(entry_a or {"total_units": None, "avg_daily_units": None})},
        "category_b": {"name": category_b, **(entry_b or {"total_units": None, "avg_daily_units": None})},
        "leader": leader,
    }


def get_store_performance(store_id: str, recent_days: int = 30, baseline_days: int = 180) -> dict[str, Any]:
    """One store's recent average daily units sold vs. its own longer-run baseline."""

    row = db.fetch_store_performance(store_id, recent_days=recent_days, baseline_days=baseline_days)
    if row is None:
        return {"store_id": store_id, "store_name": None, "recent_avg": None, "baseline_avg": None, "pct_change": None}

    recent_avg = round(row["recent_avg"], 2) if row["recent_avg"] is not None else None
    baseline_avg = round(row["baseline_avg"], 2) if row["baseline_avg"] is not None else None

    return {
        "store_id": store_id,
        "store_name": row["store_name"],
        "recent_days": recent_days,
        "baseline_days": baseline_days,
        "recent_avg": recent_avg,
        "baseline_avg": baseline_avg,
        "pct_change": _pct_change(recent_avg, baseline_avg),
    }


def get_trending_down(
    level: str, threshold_pct: float = DEFAULT_TREND_THRESHOLD_PCT, connection=None
) -> dict[str, Any]:
    """Products or stores whose latest month-over-month growth is at/below threshold_pct."""

    if level not in ("product", "store"):
        raise ValueError('level must be "product" or "store"')

    rows = db.fetch_trending_down(level, threshold_pct=threshold_pct, connection=connection)
    id_key = "product_id" if level == "product" else "store_id"
    name_key = "product_name" if level == "product" else "store_name"

    items = [
        {
            "id": row[id_key],
            "name": row[name_key],
            "mom_growth_pct": round(row["avg_mom_growth_pct"], 2),
        }
        for row in rows
    ]

    return {"level": level, "threshold_pct": threshold_pct, "count": len(items), "items": items}


def get_predicted_revenue_summary(connection=None) -> dict[str, Any]:
    """Total predicted units/revenue across the whole 30-day forecast.

    Revenue is approximate: forecast_results has no price of its own, so
    each product's base_price is used as a proxy for its future price.
    """

    row = db.fetch_predicted_revenue(connection=connection)
    if row is None or row["total_predicted_units"] is None:
        return {"total_predicted_units": None, "total_predicted_revenue": None}

    return {
        "total_predicted_units": round(row["total_predicted_units"], 2),
        "total_predicted_revenue": round(row["total_predicted_revenue"], 2),
        "note": "Revenue is approximate -- computed from each product's base_price, not a simulated future price.",
    }


def get_kpi_summary() -> dict[str, Any]:
    """The dashboard's at-a-glance KPI numbers, built from the same functions above.

    Opening a Databricks connection costs several seconds on its own; this
    reuses ONE connection for all 5 underlying queries instead of paying
    that cost 5 times, which is what made /kpis slow enough to trip the
    frontend's request timeout.
    """

    with db.connection_scope() as connection:
        top_category = get_top_category(connection=connection)
        top_product = get_top_product(connection=connection)
        revenue = get_predicted_revenue_summary(connection=connection)
        trending_products = get_trending_down("product", connection=connection)
        trending_stores = get_trending_down("store", connection=connection)

    return {
        "top_category": top_category,
        "top_product": top_product,
        "predicted_revenue": revenue,
        "at_risk_count": trending_products["count"] + trending_stores["count"],
    }
