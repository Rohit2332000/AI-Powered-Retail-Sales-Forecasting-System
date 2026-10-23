# API Testing Examples (Swagger / `/docs`)

Copy-paste examples for every endpoint in [main.py](main.py), for testing via the
Swagger UI at `http://127.0.0.1:8000/docs`.

All IDs below are real rows pulled from the Databricks tables (not made up):

| product_id | product_name | category |
|---|---|---|
| P001 | Voltix Smart Watch Pro 300 | Electronics |
| P002 | FreshFarm Packaged Foods Essential 400 | Grocery |
| P003 | Urbane Women's Classic 700 | Clothing |
| P004 | Coolwell Air Conditioner Max 500 | Home Appliances |
| P005 | Peakform Fitness Equipment Elite 200 | Sports & Fitness |
| P006 | Luvena Skincare Daily 900 | Beauty & Personal Care |

| store_id | store_name | city | region |
|---|---|---|---|
| S01 | Mumbai Flagship 01 | Mumbai | West |
| S02 | Bengaluru Flagship 02 | Bengaluru | South |
| S03 | Delhi Standard 03 | Delhi | North |

> Tip: In Swagger, click **"Try it out"** on any endpoint, paste the query params
> or JSON body below, then click **Execute**.

---

## GET /health

No parameters. Confirms the API is up and can reach the Databricks SQL warehouse.

```
GET /health
```

---

## GET /products

No parameters. Returns every product (for populating a product dropdown).

```
GET /products
```

---

## GET /stores

No parameters. Returns every store (for populating a store dropdown).

```
GET /stores
```

---

## GET /forecast

Three ways to call it — paste these into the query params fields in Swagger.

**1. All forecasts for one product + store:**
```
product_id = P004
store_id   = S01
```

**2. Filter by category only (no product/store):**
```
category = Home Appliances
```

**3. Merged actual + predicted trend for one series** (requires both `product_id`
and `store_id`; this is the mode the frontend chart uses):
```
product_id      = P004
store_id        = S01
include_history = true
history_days    = 90
```

---

## GET /forecast/trend/all

No parameters. Dumps every forecast row across all products/stores, grouped
by product — a full unrestricted feed.

```
GET /forecast/trend/all
```

---

## POST /forecast/explain

GenAI explanation of an already-computed forecast (never generates numbers).
Body (JSON):

```json
{
  "product_id": "P004",
  "store_id": "S01",
  "user_question": "Why is demand for this air conditioner rising heading into next month?"
}
```

`user_question` is optional — omit it (or send `null`) for a general explanation:

```json
{
  "product_id": "P002",
  "store_id": "S02"
}
```

---

## POST /chat

Free-text question, answered by an LLM using real SQL tool calls (never
invents numbers). `history` and `store_id`/`store_name` are optional.

```json
{
  "message": "Which category is selling the most right now?",
  "history": [],
  "store_id": "S02",
  "store_name": "Bengaluru Flagship 02"
}
```

Follow-up turn, showing `history` in use:

```json
{
  "message": "And which specific product is driving that?",
  "history": [
    { "role": "user", "content": "Which category is selling the most right now?" },
    { "role": "assistant", "content": "Electronics is the top-selling category at this store, led mainly by P001." }
  ]
}
```

> Note: the chat LLM only has the tools listed below (top category/product,
> compare categories, store performance, trending down, predicted revenue) —
> it's instructed to say "I don't have a tool for that" rather than guess if
> you ask something outside that set (e.g. "what's trending **up**?" has no
> matching tool, only `get_trending_down` exists).

---

## POST /chat/quick

Deterministic, no-LLM answers to fixed buttons. `action` + `params`:

**top_category** (no params needed):
```json
{
  "action": "top_category",
  "params": {}
}
```

**top_product**, optionally scoped to one store:
```json
{
  "action": "top_product",
  "params": { "store_id": "S02" }
}
```

**compare_categories** (requires `category_a` + `category_b`):
```json
{
  "action": "compare_categories",
  "params": { "category_a": "Electronics", "category_b": "Grocery" }
}
```

**store_performance** (requires `store_id`):
```json
{
  "action": "store_performance",
  "params": { "store_id": "S01" }
}
```

**underperforming_stores** (no params needed):
```json
{
  "action": "underperforming_stores",
  "params": {}
}
```

**trending_down_products** (no params needed):
```json
{
  "action": "trending_down_products",
  "params": {}
}
```

**predicted_revenue** (no params needed):
```json
{
  "action": "predicted_revenue",
  "params": {}
}
```

---

## GET /kpis

No parameters. Dashboard header numbers (top category, top product, at-risk
count, predicted revenue).

```
GET /kpis
```

---

## POST /pipeline/trigger

No body. Re-runs the pipeline job against whatever's already in the volume.

```
POST /pipeline/trigger
```

Response gives you a `run_id` — copy it for the next call.

---

## GET /pipeline/status/{run_id}

Path parameter — paste the `run_id` returned by `/pipeline/trigger` or
`/pipeline/upload-and-trigger`:

```
run_id = 123456
```

---

## POST /pipeline/upload-and-trigger

Upload endpoint — matches the UI's "Upload New Data & Retrain" button. In
Swagger this renders as **two separate "Choose File" pickers**,
`sales_transactions_file` and `calendar_marketing_file`. Both are optional,
but at least one is required. Filenames **must** start with the matching
prefix and end in `.csv`, e.g.:

- `sales_transactions_file` → `sales_transactions_2026_09.csv`
- `calendar_marketing_file` → `calendar_marketing_external_2026_09.csv`

Click "Try it out" → pick one or both files → Execute. Uploads to the
Volume and triggers the pipeline job in one call; response includes a
`run_id` to poll via `GET /pipeline/status/{run_id}`.

`POST /pipeline/trigger` (above, no body) is the "Retrain Model" button —
re-runs against data already uploaded, no new file needed.

---

## Suggested demo order for the video

1. `GET /health` — show it's alive and connected.
2. `GET /products` and `GET /stores` — show the real reference data.
3. `GET /forecast` with `product_id=P004&store_id=S01` — show a raw forecast.
4. `GET /forecast?include_history=true` (same product/store) — show the merged actual+predicted trend.
5. `POST /forecast/explain` — show the GenAI narrative on top of that same forecast.
6. `POST /chat/quick` (`top_category`) then `POST /chat` free-text — show deterministic vs. LLM-driven answers.
7. `GET /kpis` — tie it back to a dashboard view.
