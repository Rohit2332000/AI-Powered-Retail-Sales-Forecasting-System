"""FastAPI microservice serving retail sales forecasts from Databricks."""

import logging
import os
from datetime import datetime
from typing import List, Optional, Union

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.data import analytics, databricks_jobs
from src.data.databricks_jobs import DatabricksJobsConfigError
from src.data.db import (
    check_connection,
    fetch_distinct_products,
    fetch_forecast_predictions,
    fetch_forecasts,
    fetch_history_actuals,
    fetch_product_static,
    fetch_products,
    fetch_recent_sales_features,
    fetch_stores,
)
from src.genai.chatbot import QUICK_ACTIONS, run_chat, run_quick_action
from src.genai.explain_context import build_forecast_context
from src.genai.forecast_explainer import generate_explanation
from src.genai.llm_client import LLMConfigError
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    ExplainRequest,
    ExplainResponse,
    ForecastListResponse,
    ForecastPoint,
    ForecastRecord,
    ForecastSummary,
    ForecastTrendResponse,
    HealthResponse,
    KPISummary,
    PipelineStatusResponse,
    PipelineTriggerResponse,
    PipelineUploadAndTriggerResponse,
    PredictedRevenueStat,
    Product,
    ProductForecastGroup,
    QuickActionRequest,
    Store,
    TopCategoryStat,
    TopProductStat,
    UploadedFileInfo,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales_forecasting_api")

app = FastAPI(
    title="Sales Forecasting API",
    description="Serves retail sales forecast data from the Databricks gold layer (sales_ai.gold.forecast_results).",
    version="0.1.0",
)

# Allows the separately-deployed frontend (Vite dev server locally, or a
# different origin/domain in production) to call this API from the browser.
# Configurable via CORS_ALLOW_ORIGINS (comma-separated) since frontend and
# backend can be deployed to different hosts.
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Confirms the API is running and can reach the Databricks SQL warehouse.",
)
def health() -> HealthResponse:
    try:
        check_connection()
    except Exception:
        logger.exception("Databricks health check failed")
        raise HTTPException(
            status_code=500,
            detail="API is up but the Databricks connection check failed. Check server logs for details.",
        )

    return HealthResponse(status="okay I'm, ready to serve you!!", databricks_connected=True)


def _to_date(value):
    """Normalize a date/datetime value from the connector to a plain date.

    forecast_date comes back as a TIMESTAMP (datetime) while sales_features.date
    comes back as a DATE (date) -- both are normalized here so they compare and
    sort correctly as dict keys.
    """

    return value.date() if isinstance(value, datetime) else value


def _merge_points(history_rows: list[dict], forecast_rows: list[dict]) -> List[ForecastPoint]:
    """Merge sales_features actuals and forecast_results predictions into one
    chronological timeline, keyed by date."""

    points: dict = {}

    for row in history_rows:
        row_date = _to_date(row["date"])
        points[row_date] = ForecastPoint(date=row_date, actual_units=row["units_sold"], predicted_units=None)

    for row in forecast_rows:
        forecast_date = _to_date(row["forecast_date"])
        existing = points.get(forecast_date)
        if existing is not None:
            existing.predicted_units = row["predicted_units"]
        else:
            points[forecast_date] = ForecastPoint(
                date=forecast_date, actual_units=None, predicted_units=row["predicted_units"]
            )

    return [points[key] for key in sorted(points)]


@app.get(
    "/forecast",
    response_model=Union[ForecastListResponse, ForecastTrendResponse],
    summary="Get sales forecasts (flat list, or merged actual+forecast trend)",
    description=(
        "Two modes, selected by `include_history`:\n\n"
        "- **include_history=false (default)** — unchanged: returns forecast rows as-is "
        "from sales_ai.gold.forecast_results, filtered by product_id/store_id/category. "
        "Response shape: `{count, results}`.\n\n"
        "- **include_history=true** — merges real history with the model's forecast into "
        "one chronological timeline for a single series. **actual_units is sourced from "
        "sales_ai.gold.sales_features; predicted_units is sourced from "
        "sales_ai.gold.forecast_results.** Requires both product_id and store_id (400 if "
        "either is missing — the merge only makes sense for one series at a time). "
        "Response shape: `{product_id, store_id, history_days, points}`, where each point "
        "has `actual_units` (past dates) or `predicted_units` (future dates), with the "
        "other null."
    ),
)
def get_forecast(
    product_id: Optional[str] = Query(None, description="Filter by product ID"),
    store_id: Optional[str] = Query(None, description="Filter by store ID"),
    category: Optional[str] = Query(None, description="Filter by product category (ignored when include_history=true)"),
    include_history: bool = Query(False, description="Merge in historical actuals alongside the forecast"),
    history_days: int = Query(
        90, ge=1, description="Days of history to include; only used when include_history=true"
    ),
) -> Union[ForecastListResponse, ForecastTrendResponse]:
    if include_history:
        if not product_id or not store_id:
            raise HTTPException(
                status_code=400,
                detail="include_history=true requires both product_id and store_id.",
            )

        try:
            history_rows = fetch_history_actuals(product_id, store_id, history_days)
            forecast_rows = fetch_forecast_predictions(product_id, store_id)
        except Exception:
            logger.exception("Failed to build forecast trend")
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch trend data from Databricks. Check server logs for details.",
            )

        return ForecastTrendResponse(
            product_id=product_id,
            store_id=store_id,
            history_days=history_days,
            points=_merge_points(history_rows, forecast_rows),
        )

    try:
        rows = fetch_forecasts(product_id=product_id, store_id=store_id, category=category)
    except Exception:
        logger.exception("Failed to query forecast_results")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch forecast data from Databricks. Check server logs for details.",
        )

    records = [ForecastRecord(**row) for row in rows]
    return ForecastListResponse(count=len(records), results=records)


@app.get(
    "/forecast/trend/all",
    response_model=List[ProductForecastGroup],
    summary="Get every predicted record, grouped by product",
    description=(
        "Returns every row from sales_ai.gold.forecast_results (all products, all "
        "stores, all dates), grouped by product_id. No query parameters, no filtering — "
        "a full, unrestricted dump for the multi-product view."
    ),
)
def get_forecast_trend_all() -> List[ProductForecastGroup]:
    try:
        products = fetch_distinct_products()
        groups = [
            ProductForecastGroup(
                product_id=product["product_id"],
                category=product["category"],
                records=[ForecastRecord(**row) for row in fetch_forecasts(product_id=product["product_id"])],
            )
            for product in products
        ]
    except Exception:
        logger.exception("Failed to fetch all forecast records")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch forecast data from Databricks. Check server logs for details.",
        )

    return groups


@app.post(
    "/forecast/explain",
    response_model=ExplainResponse,
    summary="Explain an existing forecast in plain language (never generates one)",
    description=(
        "**This endpoint explains a forecast that has ALREADY been computed by the ML "
        "pipeline — it never generates or alters the predicted numbers.**\n\n"
        "Step 1 computes a structured context dict directly in Python from "
        "sales_ai.gold.forecast_results (the 30-day forecast), sales_ai.gold.sales_features "
        "(last 90 days of actuals + engineered features), and sales_ai.silver.products "
        "(static product attributes). That dict is the *only* information given to the "
        "LLM — it is never shown raw table dumps.\n\n"
        "`forecast_summary` in the response always reflects that computed context, never "
        "anything the LLM says; the LLM only supplies the narrative fields "
        "(`explanation`, `key_drivers`, `risks_or_opportunities`, `recommendation`). If "
        "`user_question` is provided, the narrative answers it using only that same "
        "context; otherwise a general explanation is produced."
    ),
)
def explain_forecast(request: ExplainRequest) -> ExplainResponse:
    try:
        forecast_rows = fetch_forecasts(product_id=request.product_id, store_id=request.store_id)
    except Exception:
        logger.exception("Failed to fetch forecast_results for explanation")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch forecast data from Databricks. Check server logs for details.",
        )

    if not forecast_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for product_id={request.product_id!r}, store_id={request.store_id!r}.",
        )

    try:
        recent_rows = fetch_recent_sales_features(request.product_id, request.store_id, days=90)
    except Exception:
        logger.exception("Failed to fetch sales_features history for explanation")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch historical data from Databricks. Check server logs for details.",
        )

    if not recent_rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No historical data found for product_id={request.product_id!r}, "
                f"store_id={request.store_id!r}."
            ),
        )

    try:
        product_static = fetch_product_static(request.product_id)
    except Exception:
        logger.exception("Failed to fetch product attributes for explanation")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch product attributes from Databricks. Check server logs for details.",
        )

    context = build_forecast_context(
        product_id=request.product_id,
        store_id=request.store_id,
        forecast_rows=forecast_rows,
        recent_features_rows=recent_rows,
        product_static=product_static,
    )

    try:
        narrative = generate_explanation(context, request.user_question)
    except LLMConfigError as exc:
        logger.error("Azure OpenAI is not configured: %s", exc)
        raise HTTPException(status_code=503, detail=f"GenAI explanation service is not configured: {exc}")
    except Exception:
        logger.exception("Azure OpenAI call failed")
        raise HTTPException(
            status_code=503,
            detail="GenAI explanation service is currently unreachable. Please try again shortly.",
        )

    return ExplainResponse(
        product_id=request.product_id,
        store_id=request.store_id,
        forecast_summary=ForecastSummary(
            avg_predicted_units=context["avg_predicted_units"],
            total_predicted_units=context["total_predicted_units"],
            vs_recent_history_pct_change=context["vs_recent_history_pct_change"],
        ),
        explanation=narrative["explanation"],
        key_drivers=narrative["key_drivers"],
        risks_or_opportunities=narrative["risks_or_opportunities"],
        recommendation=narrative["recommendation"],
        model_version=str(context["model_version"]),
    )


@app.get(
    "/products",
    response_model=List[Product],
    summary="List all products",
    description="Returns every product from sales_ai.silver.products, for populating a product selector.",
)
def get_products() -> List[Product]:
    try:
        rows = fetch_products()
    except Exception:
        logger.exception("Failed to fetch products")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch products from Databricks. Check server logs for details.",
        )

    return [Product(**row) for row in rows]


@app.get(
    "/stores",
    response_model=List[Store],
    summary="List all stores",
    description="Returns every store from sales_ai.silver.stores, for populating a store selector.",
)
def get_stores() -> List[Store]:
    try:
        rows = fetch_stores()
    except Exception:
        logger.exception("Failed to fetch stores")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch stores from Databricks. Check server logs for details.",
        )

    return [Store(**row) for row in rows]


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a free-text business question (GenAI, tool-calling)",
    description=(
        "General-purpose analytics chatbot, separate from /forecast/explain. The LLM is "
        "given a set of tools backed by real SQL queries (analytics.py) and can only "
        "narrate whatever those tools actually return — it never invents numbers, "
        "products, categories, or stores. Optionally pass `history` (prior user/assistant "
        "turns) for conversational context. Response shape matches POST /chat/quick, so "
        "both can render in the same chat thread."
    ),
)
def chat(request: ChatRequest) -> ChatResponse:
    history = [{"role": turn.role, "content": turn.content} for turn in request.history]

    try:
        result = run_chat(
            request.message,
            history=history,
            store_id=request.store_id,
            store_name=request.store_name,
        )
    except LLMConfigError as exc:
        logger.error("Azure OpenAI is not configured: %s", exc)
        raise HTTPException(status_code=503, detail=f"Chat service is not configured: {exc}")
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=503,
            detail="Chat service is currently unreachable. Please try again shortly.",
        )

    return ChatResponse(**result)


@app.post(
    "/chat/quick",
    response_model=ChatResponse,
    summary="Run a predefined quick-question button (deterministic, no LLM)",
    description=(
        "Answers one of a fixed set of business questions directly from analytics.py — "
        "no LLM call, so it's fast and always deterministic. Same response shape as "
        "POST /chat, so quick-button answers and free-text answers render in one unified "
        "conversation. Valid `action` values: " + ", ".join(sorted(QUICK_ACTIONS.keys())) + ". "
        "`compare_categories` requires params `category_a`/`category_b`; "
        "`store_performance` requires params `store_id`."
    ),
)
def chat_quick(request: QuickActionRequest) -> ChatResponse:
    try:
        result = run_quick_action(request.action, request.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing required param: {exc}")
    except Exception:
        logger.exception("Quick action failed: action=%s params=%s", request.action, request.params)
        raise HTTPException(
            status_code=500,
            detail="Failed to compute that answer from Databricks. Check server logs for details.",
        )

    return ChatResponse(**result)


@app.get(
    "/kpis",
    response_model=KPISummary,
    summary="Dashboard KPI summary",
    description=(
        "At-a-glance numbers for the dashboard header: top category, top product, "
        "at-risk (trending down) product+store count, and approximate total predicted "
        "revenue. Reuses the same analytics.py functions as /chat and /chat/quick."
    ),
)
def get_kpis() -> KPISummary:
    try:
        summary = analytics.get_kpi_summary()
    except Exception:
        logger.exception("Failed to compute KPI summary")
        raise HTTPException(
            status_code=500,
            detail="Failed to compute KPI summary from Databricks. Check server logs for details.",
        )

    return KPISummary(
        top_category=TopCategoryStat(**summary["top_category"]),
        top_product=TopProductStat(**summary["top_product"]),
        predicted_revenue=PredictedRevenueStat(**summary["predicted_revenue"]),
        at_risk_count=summary["at_risk_count"],
    )


ALLOWED_UPLOAD_PREFIXES = ("sales_transactions", "calendar_marketing_external")


def _validate_upload_filename(filename: Optional[str]) -> str:
    """Enforce what the Bronze notebook expects: a CSV whose filename starts
    with one of the recognized prefixes. Raises HTTPException(400) otherwise."""

    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    if not filename.startswith(ALLOWED_UPLOAD_PREFIXES):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Filename must start with one of {ALLOWED_UPLOAD_PREFIXES} -- "
                "that's what the Bronze notebook expects."
            ),
        )
    return filename


async def _upload_pipeline_files(files: List[UploadFile]) -> List[UploadedFileInfo]:
    """Upload logic for /pipeline/upload-and-trigger.

    Supports one or more files in a single request (e.g. a sales_transactions
    file and a calendar_marketing_external file together, or just one --
    whichever data is available). Every filename is validated up front, before
    anything is uploaded, so a bad file in the batch fails the whole request
    rather than leaving a partial upload. Raises HTTPException on failure.
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    for file in files:
        _validate_upload_filename(file.filename)

    uploaded: List[UploadedFileInfo] = []
    for file in files:
        content = await file.read()
        try:
            volume_path = databricks_jobs.upload_file_to_volume(file.filename, content)
        except DatabricksJobsConfigError as exc:
            logger.error("Pipeline upload is not configured: %s", exc)
            raise HTTPException(status_code=503, detail=f"Pipeline upload is not configured: {exc}")
        except Exception:
            logger.exception("Failed to upload pipeline file %s to Databricks Volume", file.filename)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to upload {file.filename} to Databricks. Check server logs for details.",
            )
        uploaded.append(UploadedFileInfo(filename=file.filename, volume_path=volume_path))

    return uploaded


def _trigger_pipeline_job() -> int:
    """Shared trigger logic for /pipeline/trigger and /pipeline/upload-and-trigger.
    Returns the new run_id. Raises HTTPException on failure."""

    try:
        return databricks_jobs.trigger_job()
    except DatabricksJobsConfigError as exc:
        logger.error("Pipeline trigger is not configured: %s", exc)
        raise HTTPException(status_code=503, detail=f"Pipeline trigger is not configured: {exc}")
    except Exception:
        logger.exception("Failed to trigger Databricks job")
        raise HTTPException(
            status_code=502,
            detail="Failed to trigger the pipeline job. Check server logs for details.",
        )


@app.post(
    "/pipeline/trigger",
    response_model=PipelineTriggerResponse,
    summary="Trigger the data pipeline job (run-now, no new file)",
    description=(
        "Triggers a run-now on DATABRICKS_JOB_ID against whatever files are already in "
        "the Volume -- no upload needed. Use this to re-run the pipeline without "
        "supplying new data. Returns immediately with the new run_id; poll "
        "GET /pipeline/status/{run_id} for progress."
    ),
)
def trigger_pipeline() -> PipelineTriggerResponse:
    run_id = _trigger_pipeline_job()
    return PipelineTriggerResponse(run_id=run_id)


@app.get(
    "/pipeline/status/{run_id}",
    response_model=PipelineStatusResponse,
    summary="Check a pipeline run's status",
    description=(
        "Returns the run's life_cycle_state (e.g. PENDING, RUNNING, TERMINATED) and, once "
        "TERMINATED, its result_state (e.g. SUCCESS, FAILED) and state_message. Meant to "
        "be polled every few seconds by the frontend until the run reaches a terminal state."
    ),
)
def get_pipeline_status(run_id: int) -> PipelineStatusResponse:
    try:
        status = databricks_jobs.get_run_status(run_id)
    except DatabricksJobsConfigError as exc:
        logger.error("Pipeline status check is not configured: %s", exc)
        raise HTTPException(status_code=503, detail=f"Pipeline status check is not configured: {exc}")
    except Exception:
        logger.exception("Failed to fetch run status for run_id=%s", run_id)
        raise HTTPException(
            status_code=502,
            detail="Failed to fetch pipeline run status. Check server logs for details.",
        )

    return PipelineStatusResponse(run_id=run_id, **status)


@app.post(
    "/pipeline/upload-and-trigger",
    response_model=PipelineUploadAndTriggerResponse,
    summary="Upload raw data file(s) and immediately trigger the pipeline",
    description=(
        "Uploads one or both files to DATABRICKS_VOLUME_INCOMING_PATH via the Databricks "
        "Files API (overwriting any existing file with the same name), then triggers a "
        "run-now on DATABRICKS_JOB_ID. Provide a sales_transactions file and/or a "
        "calendar_marketing_external file -- whichever data is available; at least one is "
        "required. Every filename must start with `sales_transactions` or "
        "`calendar_marketing_external` respectively; if either fails validation, nothing "
        "is uploaded. Returns the run_id immediately; poll GET /pipeline/status/{run_id} "
        "for progress. Use POST /pipeline/trigger instead if you want to re-run against "
        "data that's already uploaded, with no new file."
    ),
)
async def upload_and_trigger_pipeline(
    sales_transactions_file: Optional[UploadFile] = File(
        None, description="CSV whose filename starts with 'sales_transactions', e.g. sales_transactions_2026_09.csv"
    ),
    calendar_marketing_file: Optional[UploadFile] = File(
        None,
        description="CSV whose filename starts with 'calendar_marketing_external', e.g. calendar_marketing_external_2026_09.csv",
    ),
) -> PipelineUploadAndTriggerResponse:
    files = [f for f in (sales_transactions_file, calendar_marketing_file) if f is not None]
    uploaded = await _upload_pipeline_files(files)
    run_id = _trigger_pipeline_job()
    return PipelineUploadAndTriggerResponse(files=uploaded, run_id=run_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
