"""LLM tool-calling orchestration for POST /chat.

Mirrors forecast_explainer.py's role for /forecast/explain: owns prompt
construction and response guardrails (parse/validate, retry once, fall back
to a templated answer built from raw data) so a chat request never fails
outright just because the LLM's final answer didn't come back as valid
JSON. The one addition here is the tool-calling round trip itself -- the
LLM chooses which analytics.py function(s) to call, this module executes
them against real Databricks-backed data, and the LLM narrates only what
those functions actually returned.
"""

import json
import logging
import re
from typing import Any, Optional

from src.data import analytics
from src.genai.llm_client import call_chat

logger = logging.getLogger("sales_forecasting_api.chatbot")

SYSTEM_PROMPT = """You are a retail sales analytics assistant for a sales forecasting dashboard.

Answer questions using ONLY the tools provided -- call a tool to fetch real
data, then narrate exactly what the tool(s) returned. Never invent numbers,
product names, categories, or store names that are not present in a tool
result. If no tool fits the question, or a tool didn't return enough
information to answer, say so rather than guessing.

If the user asks to compare two categories but only names one (or none),
ask them to clarify instead of guessing at the second category.

Once you have the data you need, respond with a single JSON object of this
exact shape and nothing else:
{"answer": "plain language answer, may use simple markdown"}"""


def _build_system_prompt(store_id: Optional[str], store_name: Optional[str]) -> str:
    if not store_id:
        return SYSTEM_PROMPT

    label = store_name or store_id
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"The user currently has store {store_id} ({label}) selected on the dashboard. "
        f'When calling get_top_category or get_top_product, pass store_id="{store_id}" unless '
        f"the question explicitly asks about all stores or names a different store."
    )

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_top_category",
            "description": (
                "Get the best-selling product category by total actual units sold over a recent "
                "window, optionally scoped to a single store."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Lookback window in days (default 90)"},
                    "store_id": {
                        "type": "string",
                        "description": "Optional store ID (e.g. S01) to scope to one store. Omit for all stores.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_product",
            "description": (
                "Get the best-selling individual product by total actual units sold over a recent "
                "window, optionally scoped to a single store."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Lookback window in days (default 90)"},
                    "store_id": {
                        "type": "string",
                        "description": "Optional store ID (e.g. S01) to scope to one store. Omit for all stores.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_categories",
            "description": "Compare total and average daily actual units sold between two product categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_a": {"type": "string", "description": "First category name"},
                    "category_b": {"type": "string", "description": "Second category name"},
                    "days": {"type": "integer", "description": "Lookback window in days (default 90)"},
                },
                "required": ["category_a", "category_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_store_performance",
            "description": (
                "Compare one store's recent average daily units sold against its own "
                "longer-run historical baseline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "Store ID, e.g. S01"},
                    "recent_days": {"type": "integer", "description": "Recent window in days (default 30)"},
                    "baseline_days": {"type": "integer", "description": "Baseline window in days (default 180)"},
                },
                "required": ["store_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trending_down",
            "description": (
                "List products or stores whose latest month-over-month growth is at or "
                "below a negative threshold (i.e. trending down / underperforming)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["product", "store"],
                        "description": "Whether to check products or stores",
                    },
                    "threshold_pct": {
                        "type": "number",
                        "description": "Growth % threshold, default -10 (i.e. down 10% or more)",
                    },
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_predicted_revenue_summary",
            "description": "Get total predicted units and approximate predicted revenue across the entire 30-day forecast.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_DISPATCH = {
    "get_top_category": lambda args: analytics.get_top_category(
        days=args.get("days") or 90, store_id=args.get("store_id")
    ),
    "get_top_product": lambda args: analytics.get_top_product(
        days=args.get("days") or 90, store_id=args.get("store_id")
    ),
    "compare_categories": lambda args: analytics.compare_categories(
        args["category_a"], args["category_b"], days=args.get("days") or 90
    ),
    "get_store_performance": lambda args: analytics.get_store_performance(
        args["store_id"],
        recent_days=args.get("recent_days") or 30,
        baseline_days=args.get("baseline_days") or 180,
    ),
    "get_trending_down": lambda args: analytics.get_trending_down(
        args["level"],
        threshold_pct=args["threshold_pct"] if args.get("threshold_pct") is not None else -10.0,
    ),
    "get_predicted_revenue_summary": lambda args: analytics.get_predicted_revenue_summary(),
}

_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


def _parse_answer(raw_content: Optional[str]) -> Optional[str]:
    if not raw_content:
        return None
    try:
        parsed = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
        return None
    return parsed["answer"]


def _numeric_set(text: str) -> set[float]:
    """Extract numbers as floats (not raw strings), so "85438" and "85438.0"
    compare equal instead of being treated as different tokens."""

    values = set()
    for match in _NUMBER_PATTERN.findall(text):
        try:
            values.add(float(match))
        except ValueError:
            continue
    return values


def _check_numeric_grounding(answer: str, tool_results: dict[str, Any]) -> None:
    """Log a warning (never raises) if the answer contains a number that
    doesn't appear anywhere in the tool results actually returned. Approximate
    on purpose -- same style as forecast_explainer.py's grounding check."""

    known_numbers = _numeric_set(json.dumps(tool_results, default=str))
    answer_numbers = _numeric_set(answer.replace(",", ""))
    ungrounded = answer_numbers - known_numbers
    if ungrounded:
        logger.warning("Chat answer contains numbers not found in tool results: %s", sorted(ungrounded))


def _fallback_answer(tool_results: dict[str, Any]) -> str:
    """Generic templated answer built purely from tool_results, used when the
    LLM's final JSON can't be parsed/validated even after a retry -- a chat
    request must never fail outright just because the LLM misbehaved."""

    if not tool_results:
        return "I wasn't able to find data to answer that question. Try one of the quick-question buttons instead."

    parts = [f"{name}: {json.dumps(result, default=str)}" for name, result in tool_results.items()]
    return "Here's the data I found: " + " | ".join(parts)


def _execute_tool_calls(tool_calls) -> dict[str, Any]:
    """Run each tool the model requested, returning {tool_name: result}."""

    results: dict[str, Any] = {}
    for call in tool_calls:
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        handler = TOOL_DISPATCH.get(name)
        if handler is None:
            results[name] = {"error": f"Unknown tool: {name}"}
            continue

        try:
            results[name] = handler(args)
        except Exception:
            logger.exception("Analytics function %s failed for args=%s", name, args)
            results[name] = {"error": f"{name} failed to run."}

    return results


def run_chat(
    message: str,
    history: Optional[list[dict[str, str]]] = None,
    store_id: Optional[str] = None,
    store_name: Optional[str] = None,
) -> dict[str, Any]:
    """Answer a free-text question via LLM tool-calling.

    store_id/store_name, when given (the dashboard's currently-selected
    store), are added to the system prompt as context so store-scoped tools
    apply that filter by default -- without this, get_top_category/
    get_top_product always answer globally regardless of what's selected.

    Returns {"answer": str, "data": dict, "functions_used": list[str]}.
    Never raises for a malformed final answer -- falls back to a templated
    one built from the actual tool results instead. Network/credential
    failures from the LLM call itself DO propagate, so the caller can turn
    them into a clean 503 (same contract as
    forecast_explainer.generate_explanation).
    """

    messages: list[dict[str, Any]] = [{"role": "system", "content": _build_system_prompt(store_id, store_name)}]
    for turn in history or []:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    assistant_message = call_chat(messages, tools=TOOLS)
    tool_results: dict[str, Any] = {}

    if assistant_message.tool_calls:
        messages.append(assistant_message.model_dump(exclude_none=True))
        tool_results = _execute_tool_calls(assistant_message.tool_calls)
        for call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(tool_results.get(call.function.name, {}), default=str),
                }
            )

    for attempt in range(2):
        final_message = call_chat(messages, json_mode=True)
        answer = _parse_answer(final_message.content)
        if answer is not None:
            if tool_results:
                _check_numeric_grounding(answer, tool_results)
            return {"answer": answer, "data": tool_results, "functions_used": list(tool_results.keys())}
        logger.warning("Chat final answer failed validation on attempt %d: %r", attempt + 1, final_message.content)

    logger.warning("Chat final answer failed validation twice; falling back to templated answer.")
    return {
        "answer": _fallback_answer(tool_results),
        "data": tool_results,
        "functions_used": list(tool_results.keys()),
    }


# ---------------------------------------------------------------------------
# Quick-question buttons: deterministic answers, same analytics.py functions,
# no LLM call -- fast and always consistent, for the frontend's predefined
# question buttons. Response shape matches run_chat()'s so both paths render
# in the same chat thread.
# ---------------------------------------------------------------------------


def _scope_phrase(result: dict[str, Any]) -> str:
    return f" at store {result['store_id']}" if result.get("store_id") else " across all stores"


def _format_top_category(result: dict[str, Any]) -> str:
    if result["category"] is None:
        return "No sales data is available for that window."
    return (
        f"**{result['category']}** is the top-selling category{_scope_phrase(result)} over the last "
        f"{result['days']} days, with {result['total_units']:,.0f} units sold."
    )


def _format_top_product(result: dict[str, Any]) -> str:
    if result["product_id"] is None:
        return "No sales data is available for that window."
    return (
        f"**{result['product_name']}** ({result['product_id']}) is the top-selling product"
        f"{_scope_phrase(result)} over the last {result['days']} days, with {result['total_units']:,.0f} units sold."
    )


def _format_compare_categories(result: dict[str, Any]) -> str:
    entry_a, entry_b = result["category_a"], result["category_b"]
    if entry_a["total_units"] is None or entry_b["total_units"] is None:
        return "Could not find data for one or both of those categories."
    return (
        f"Over the last {result['days']} days: **{entry_a['name']}** sold {entry_a['total_units']:,.0f} units "
        f"({entry_a['avg_daily_units']:,.1f}/day) vs. **{entry_b['name']}** at {entry_b['total_units']:,.0f} units "
        f"({entry_b['avg_daily_units']:,.1f}/day). **{result['leader']}** leads."
    )


def _format_store_performance(result: dict[str, Any]) -> str:
    if result["recent_avg"] is None:
        return f"No data found for store {result['store_id']}."
    if result["pct_change"] is None:
        return (
            f"**{result['store_name']}** is averaging {result['recent_avg']:,.1f} units/day over the "
            f"last {result['recent_days']} days (no baseline available for comparison)."
        )
    direction = "up" if result["pct_change"] >= 0 else "down"
    return (
        f"**{result['store_name']}** is averaging {result['recent_avg']:,.1f} units/day over the last "
        f"{result['recent_days']} days, {direction} {abs(result['pct_change']):.1f}% versus its "
        f"{result['baseline_days']}-day baseline of {result['baseline_avg']:,.1f} units/day."
    )


def _format_trending_down(result: dict[str, Any]) -> str:
    label = "products" if result["level"] == "product" else "stores"
    if result["count"] == 0:
        return f"No {label} are trending down beyond {result['threshold_pct']}% month-over-month."
    lines = "\n".join(f"- {item['name']}: {item['mom_growth_pct']:+.1f}%" for item in result["items"])
    return (
        f"{result['count']} {label} are trending down (month-over-month at or below "
        f"{result['threshold_pct']}%):\n\n{lines}"
    )


def _format_predicted_revenue(result: dict[str, Any]) -> str:
    if result["total_predicted_units"] is None:
        return "No forecast data is available."
    return (
        f"The 30-day forecast totals **{result['total_predicted_units']:,.0f} units**, for an approximate "
        f"predicted revenue of **{result['total_predicted_revenue']:,.0f}** (using each product's base price "
        f"as a stand-in for its future price)."
    )


QUICK_ACTIONS: dict[str, tuple] = {
    "top_category": (
        lambda params: analytics.get_top_category(store_id=params.get("store_id")),
        _format_top_category,
    ),
    "top_product": (
        lambda params: analytics.get_top_product(store_id=params.get("store_id")),
        _format_top_product,
    ),
    "compare_categories": (
        lambda params: analytics.compare_categories(params["category_a"], params["category_b"]),
        _format_compare_categories,
    ),
    "store_performance": (
        lambda params: analytics.get_store_performance(params["store_id"]),
        _format_store_performance,
    ),
    "underperforming_stores": (lambda params: analytics.get_trending_down("store"), _format_trending_down),
    "trending_down_products": (lambda params: analytics.get_trending_down("product"), _format_trending_down),
    "predicted_revenue": (lambda params: analytics.get_predicted_revenue_summary(), _format_predicted_revenue),
}


def run_quick_action(action: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Deterministic (no-LLM) answer for one of the predefined quick-question
    buttons -- calls the same analytics.py functions as run_chat's tools,
    just without an LLM deciding which one to call.

    Raises KeyError if a required param is missing, ValueError if the action
    is unknown -- both are caller errors (400), not server errors.
    """

    entry = QUICK_ACTIONS.get(action)
    if entry is None:
        raise ValueError(f"Unknown quick action: {action}")

    fetch, formatter = entry
    result = fetch(params or {})
    return {"answer": formatter(result), "data": {action: result}, "functions_used": [action]}
