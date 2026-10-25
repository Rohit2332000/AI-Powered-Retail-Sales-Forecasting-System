# AI-Powered Retail Sales Forecasting System — Backend

FastAPI backend for an AI-powered retail sales forecasting platform. It connects the frontend to Databricks data and ML workflows, exposes forecasting and analytics APIs, supports data uploads and pipeline triggers, and provides an LLM-powered analytics chat experience.


## Problem Statement

Retail businesses generate large volumes of sales data, but turning this data into accurate forecasts and actionable business insights can be challenging. Traditional reporting often focuses on historical performance and may not provide an integrated way to forecast future demand, understand sales trends, or respond quickly to changing business conditions.

Additionally, updating datasets, rerunning data pipelines, comparing machine learning models, and interpreting forecasting results can require multiple disconnected tools and manual effort.

## Solution

The **AI-Powered Retail Sales Forecasting System** provides an integrated platform that combines machine learning, data engineering, and generative AI to support data-driven retail decision-making.

The system addresses these challenges through:

- **30-Day Sales Forecasting:** Generates sales forecasts for the next 30 days to help businesses plan inventory, demand, and operations.
- **Automated Data Processing:** Uses Databricks and a Bronze–Silver–Gold medallion architecture to organize and transform retail data.
- **Model Experimentation and Selection:** Integrates MLflow to track experiments, compare model performance, and select the best-performing model according to configured evaluation metrics.
- **On-Demand Pipeline Execution:** Allows users to upload new datasets and trigger data processing and model workflows through backend APIs.
- **AI-Powered Analytics Chat:** Enables users to ask natural-language questions about retail sales and obtain data-driven insights through an LLM-powered interface.
- **Forecast Explanations and KPIs:** Provides API endpoints for forecasts, key performance indicators, and explanations to support business analysis.
- **LLM Observability:** Integrates LangSmith to trace and debug instrumented LLM and application workflows.
- **Automated Deployment:** Uses Docker, AWS EC2, and GitHub Actions CI/CD to package and deploy the frontend and backend as separate services.

## Business Value

The platform brings forecasting, data processing, model evaluation, and conversational analytics into one workflow. It aims to reduce manual operational effort, improve visibility into future sales demand, and help retail teams make more informed decisions using historical data and machine learning predictions.

## Architecture



<img width="1672" height="940" alt="AI-Powered Retail Forecasting Architecture" src="https://github.com/user-attachments/assets/07fe1721-d50e-4d24-a475-f1e03eb460d7" />



The platform is organized into three services:

- **Frontend:** React + Vite dashboard and analytics chat.
- **Backend:** Python + FastAPI REST API, deployed in a Docker container on its own AWS EC2 instance.
- **ML pipeline:** Databricks Workflows/Jobs for Bronze–Silver–Gold data processing, model training, evaluation, and batch inference.
- **MLflow:** Experiment tracking, metrics/parameters/artifacts comparison, model selection, and model registry/versioning within the ML workflow.
- **LangSmith:** Tracing and observability for supported LLM/application workflows.
- **CI/CD:** GitHub Actions builds, tests, packages, and deploys the frontend and backend to separate EC2 instances.

Typical flow:

1. Retail CSV files are uploaded through the frontend or backend API.
2. The backend validates the request and uses the Databricks API to upload data or trigger a workflow.
3. Databricks processes data through Bronze, Silver, and Gold layers.
4. Models are trained and evaluated; MLflow tracks experiments and supports selecting/versioning the best model according to the configured evaluation criteria.
5. Forecasts and analytics are served to the frontend through REST/JSON APIs.
6. LangSmith captures traces for instrumented LLM/application operations.

## Features

- REST API built with FastAPI.
- Retail sales forecasting and forecast filtering.
- Forecast/KPI and analytics endpoints.
- Upload data files and trigger Databricks jobs.
- Monitor pipeline execution status.
- LLM-powered analytics chat.
- Databricks SQL connectivity for data access.
- LangSmith tracing for instrumented application/LLM workflows.
- Docker-based deployment.
- GitHub Actions CI/CD support.

## Technology Stack

- **Language:** Python
- **API:** FastAPI, Uvicorn
- **Data and ML platform:** Databricks Workflows/Jobs, Databricks SQL
- **Experiment tracking/model management:** MLflow
- **LLM integration:** OpenAI-compatible client (configured through environment variables)
- **Observability:** LangSmith
- **Deployment:** Docker, AWS EC2
- **CI/CD:** GitHub Actions

## API Endpoints

The exact routes depend on the current application code. The following routes are used by this project:

| Endpoint | Purpose |
|---|---|
| `GET /healthy` | Health check, including a Databricks connection check |
| `GET /docs` | Interactive Swagger API documentation |
| `GET /kpis` | Retrieve dashboard KPI data |
| `GET /forecast` | Retrieve forecasts with optional filters/history |
| `POST /chat` | Ask questions through the analytics chat interface |
| `POST /pipeline/upload` | Upload a supported data file |
| `POST /pipeline/trigger` | Trigger the Databricks pipeline |
| `POST /pipeline/upload-and-trigger` | Upload data and trigger processing |
| `GET /pipeline/status/{run_id}` | Check the status of a pipeline run |
| `GET /forecast/explain` | Request an explanation of forecast results |

Check `/docs` in the running environment for the authoritative request schemas, parameters, and response models.

## Project Structure

Example structure (adjust names to match the repository):

```text
backend/
├── main.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── src/
    └── genai/
        └── llm_client.py
```

## Prerequisites

- Python 3.12 recommended
- pip
- Access to the required Databricks workspace, SQL warehouse, and jobs
- API credentials for the configured LLM provider
- LangSmith API key if tracing is enabled
- Docker for containerized deployment

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rohit2332000/AI-Powered-Retail-Sales-Forecasting-System.git
cd AI-Powered-Retail-Sales-Forecasting-System
```

If the backend lives in a subdirectory, enter it before continuing:

```bash
cd backend
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file from `.env.example` (or create one manually). Never commit `.env` or real credentials to Git.

Example variable names (set values according to your Databricks and provider configuration):

```dotenv
# Databricks
DATABRICKS_HOST=
DATABRICKS_TOKEN=
DATABRICKS_HTTP_PATH=

# LLM provider
OPENAI_API_KEY=
OPENAI_MODEL=

# LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=retail-sales-forecasting
```

Your application may use additional environment variables for Databricks job IDs, catalog/schema/table names, SQL settings, or model configuration. Match these names to the variables read by your code. Do not paste credentials into issues, logs, or README files.

### 5. Run the API

Use the command that matches your application entry point. If `main.py` starts Uvicorn itself:

```bash
python main.py
```

Alternatively, if the FastAPI instance is named `app` in `main.py`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/healthy`

## Docker

Build the backend image from the directory containing the Dockerfile:

```bash
docker build -t sales-ai-backend .
```

Run the container on port 8000:

```bash
docker run -d \
  --name sales-ai-backend \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  sales-ai-backend
```

For PowerShell, the equivalent is:

```powershell
docker run -d --name sales-ai-backend --restart unless-stopped -p 8000:8000 --env-file .env sales-ai-backend
```

If a container with the same name already exists, stop and remove it before recreating it:

```bash
docker stop sales-ai-backend
docker rm sales-ai-backend
```

After changing `requirements.txt`, rebuild the image and recreate the container so the new dependency is installed. If GitHub Actions handles deployment, verify that the workflow rebuilds and deploys the image.



# LIVE LINK: http://34.229.58.71:3000

## LangSmith Tracing

Set the following environment variables in the runtime environment:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=retail-sales-forecasting
```

Use a valid LangSmith API key and keep it secret. Instrument the relevant LLM/application functions using LangSmith-supported tracing (for example, `@traceable`) where appropriate. Tracing only applies to instrumented or automatically integrated operations; adding the dependency and environment variables alone does not guarantee that every API request or database call is traced.

Avoid tracing sensitive customer data or secrets. Review the LangSmith project and data-retention settings before using production data.

## GitHub Actions CI/CD

The intended deployment flow is:

1. Push a change to the backend repository.
2. GitHub Actions runs configured checks/build steps.
3. The workflow builds the Docker image.
4. The image is published or transferred using the configured workflow.
5. The workflow deploys/restarts the backend container on its dedicated EC2 instance.

Configure the repository secrets required by your actual workflow (for example, EC2 host/user, SSH private key, and environment configuration). Secret names in GitHub must match the names referenced by the workflow YAML. Never commit private keys, `.env` files, or access tokens.

## Deployment Notes

- Backend and frontend are deployed on **separate AWS EC2 instances**.
- The backend listens on port `8000` in the container.
- Configure the EC2 security group to allow port `8000` only from trusted sources where possible; avoid exposing internal services unnecessarily.
- Set the frontend's API base URL to the backend's reachable HTTP(S) address.
- Configure FastAPI CORS origins to include the actual frontend origin(s), including the production URL.
- Prefer HTTPS and a domain name for public production use. Do not rely on a changing EC2 public IP for a stable production endpoint.
- Ensure the Databricks and LLM credentials are available to the running container as environment variables.

## Troubleshooting

**`/healthy` returns HTTP 500**
- The API may be running while the Databricks SQL connection check fails.
- Verify workspace host, SQL HTTP path, token, network access, and warehouse status.
- Inspect container logs:

```bash
docker logs --tail 200 sales-ai-backend
```

**LangSmith traces are missing**
- Confirm `LANGSMITH_TRACING=true`, a valid `LANGSMITH_API_KEY`, and the intended `LANGSMITH_PROJECT`.
- Restart/recreate the container after changing environment variables.
- Confirm the relevant functions are instrumented and inspect application logs for tracing errors.

**A dependency added to `requirements.txt` is missing**
- Rebuild the Docker image and recreate the container, or verify that CI/CD rebuilt and deployed it.

**Frontend cannot call the backend**
- Confirm the backend is reachable.
- Check EC2 inbound rules, frontend API base URL, CORS configuration, and browser developer-console/network errors.

## Security

- Do not commit `.env`, API keys, Databricks tokens, or SSH private keys.
- Rotate any secret that has been accidentally exposed.
- Use least-privilege credentials and restrict inbound network access.
- Avoid logging secrets or sensitive records.




# Author

● Rohit Kumar Yadav

● GitHub: https://github.com/Rohit2332000

● LinkedIn: https://www.linkedin.com/in/rohit-kumar-yadav-42223a303/

● Portfolio: https://rohit2332000.github.io/rohityadav.github.io/

Developed as an AI-powered retail sales forecasting and analytics platform using FastAPI, Databricks, MLflow, LangSmith, Docker, AWS EC2, and GitHub Actions.
