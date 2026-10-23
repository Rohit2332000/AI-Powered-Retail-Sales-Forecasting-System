"""Pure-Python context builder for POST /forecast/explain.

Computes every fact the LLM is allowed to reference, straight from data
already fetched from the database -- no LLM call happens here. The dict this
module returns is the ONLY source of truth handed to the model; nothing in
main.py or forecast_explainer.py should pass raw table rows to the LLM
instead of this computed summary.
"""

from datetime import datetime
from statistics import mean
from typing import Any, Optional

# Same fixed-date holiday approach as reference_notebooks/07_batch_inference.py,
# since forecast dates extend past what sales_ai.gold.sales_features' historical
# calendar covers. Extend this dict as forecast dates move into new years.
FIXED_FUTURE_HOLIDAYS = {
    "2026-01-01": ("New Year", "National"),
    "2026-01-26": ("Republic Day", "National"),
}


def _to_date(value):
    """Normalize a date/datetime value from the connector to a plain date."""
    return value.date() if isinstance(value, datetime) else value


def _holidays_in_window(forecast_dates: list) -> list[dict[str, Any]]:
    holidays = []
    for raw_date in forecast_dates:
        d = _to_date(raw_date)
        name, kind = FIXED_FUTURE_HOLIDAYS.get(d.strftime("%Y-%m-%d"), (None, None))
        if name:
            holidays.append({"date": d.isoformat(), "name": name, "type": kind})
    return holidays


def build_forecast_context(
    product_id: str,
    store_id: str,
    forecast_rows: list[dict[str, Any]],
    recent_features_rows: list[dict[str, Any]],
    product_static: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the structured, computed context dict handed to the LLM.

    forecast_rows: sales_ai.gold.forecast_results rows for this product/store
    (from db.fetch_forecasts). recent_features_rows: the last-N-days
    sales_ai.gold.sales_features rows (from db.fetch_recent_sales_features).
    product_static: sales_ai.silver.products attributes (from
    db.fetch_product_static), or None if the product isn't found there.
    """

    sorted_forecast = sorted(forecast_rows, key=lambda r: _to_date(r["forecast_date"]))
    predicted_units = [float(r["predicted_units"]) for r in sorted_forecast]
    avg_predicted = mean(predicted_units)
    total_predicted = sum(predicted_units)

    sorted_history = sorted(recent_features_rows, key=lambda r: _to_date(r["date"]))
    latest = sorted_history[-1]
    rolling_mean_28 = latest.get("rolling_mean_28")

    vs_recent_history_pct_change = None
    if rolling_mean_28:
        vs_recent_history_pct_change = round((avg_predicted - rolling_mean_28) / rolling_mean_28 * 100, 2)

    daily_forecast = [
        {"date": _to_date(r["forecast_date"]).isoformat(), "predicted_units": round(float(r["predicted_units"]), 2)}
        for r in sorted_forecast
    ]

    forecast_dates = [r["forecast_date"] for r in sorted_forecast]

    return {
        "product_id": product_id,
        "store_id": store_id,
        "category": (product_static or {}).get("category") or sorted_forecast[0].get("category"),
        "lifecycle_status": (product_static or {}).get("lifecycle_status"),
        "demand_class": (product_static or {}).get("demand_class"),
        "price_elasticity_tier": (product_static or {}).get("price_elasticity_tier"),
        "forecast_horizon_days": len(sorted_forecast),
        "forecast_start_date": _to_date(forecast_dates[0]).isoformat(),
        "forecast_end_date": _to_date(forecast_dates[-1]).isoformat(),
        "avg_predicted_units": round(avg_predicted, 2),
        "total_predicted_units": round(total_predicted, 2),
        "daily_forecast": daily_forecast,
        "model_version": sorted_forecast[-1].get("model_version"),
        "recent_actuals": {
            "window_days": len(sorted_history),
            "latest_date": _to_date(latest["date"]).isoformat(),
            "latest_units_sold": latest.get("units_sold"),
            "rolling_mean_28": rolling_mean_28,
            "trend_7d": latest.get("trend_7d"),
            "trend_28d": latest.get("trend_28d"),
            "yoy_growth_pct": latest.get("yoy_growth_pct"),
            "mom_growth_pct": latest.get("mom_growth_pct"),
        },
        "vs_recent_history_pct_change": vs_recent_history_pct_change,
        "promotion": {
            "promotion_flag": bool(latest.get("promotion_flag")),
            "days_since_last_promotion": latest.get("days_since_last_promotion"),
        },
        "holidays_in_forecast_window": _holidays_in_window(forecast_dates),
    }
