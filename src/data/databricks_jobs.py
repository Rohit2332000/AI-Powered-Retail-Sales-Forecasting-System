"""Databricks Jobs + Unity Catalog Volumes integration for the data-ingestion
pipeline (upload a raw file, trigger the Bronze/Silver/Gold job, poll its run).

Reuses the same PAT token as src.data.databricks_client (the SQL warehouse
client) and llm_client.py (Azure OpenAI's env-loading pattern), but talks to
a different Databricks API surface -- Jobs + Volumes over REST via the
Databricks SDK's WorkspaceClient -- rather than SQL, so it's kept in its own
module.

DATABRICKS_HOST is read directly if set; otherwise it's derived from
DATABRICKS_SERVER_HOSTNAME (the same var the SQL client already uses), so
this works without adding a second hostname to .env.
"""

import io
import os
from pathlib import Path
from typing import Any, Optional

from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

# This file lives at backend/src/data/databricks_jobs.py -- .env is 3 levels up, at backend/.env.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST") or (
    f"https://{os.getenv('DATABRICKS_SERVER_HOSTNAME')}" if os.getenv("DATABRICKS_SERVER_HOSTNAME") else None
)
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_JOB_ID = os.getenv("DATABRICKS_JOB_ID")
DATABRICKS_VOLUME_INCOMING_PATH = os.getenv("DATABRICKS_VOLUME_INCOMING_PATH")

_client: Optional[WorkspaceClient] = None


class DatabricksJobsConfigError(RuntimeError):
    """Raised when a required Databricks Jobs/Volumes environment variable is missing."""


def get_client() -> WorkspaceClient:
    """Return a cached WorkspaceClient, or raise DatabricksJobsConfigError if unconfigured."""

    global _client

    if _client is None:
        missing = [
            name
            for name, value in {
                "DATABRICKS_HOST (or DATABRICKS_SERVER_HOSTNAME)": DATABRICKS_HOST,
                "DATABRICKS_TOKEN": DATABRICKS_TOKEN,
            }.items()
            if not value
        ]
        if missing:
            raise DatabricksJobsConfigError(f"Missing Databricks environment variable(s): {', '.join(missing)}")

        _client = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)

    return _client


def _require_job_id() -> int:
    if not DATABRICKS_JOB_ID:
        raise DatabricksJobsConfigError("Missing DATABRICKS_JOB_ID environment variable.")
    return int(DATABRICKS_JOB_ID)


def _require_volume_path() -> str:
    if not DATABRICKS_VOLUME_INCOMING_PATH:
        raise DatabricksJobsConfigError("Missing DATABRICKS_VOLUME_INCOMING_PATH environment variable.")
    return DATABRICKS_VOLUME_INCOMING_PATH.rstrip("/")


def upload_file_to_volume(filename: str, content: bytes) -> str:
    """Upload raw file bytes to DATABRICKS_VOLUME_INCOMING_PATH/filename via the
    Files API, overwriting any existing file with the same name.

    Returns the full Volume path written to.
    """

    client = get_client()
    volume_path = f"{_require_volume_path()}/{filename}"

    client.files.upload(volume_path, io.BytesIO(content), overwrite=True)
    return volume_path


def trigger_job() -> int:
    """Trigger a run-now on DATABRICKS_JOB_ID.

    Returns the new run's ID immediately -- run_now returns as soon as the
    run is queued, it does not wait for the run to finish.
    """

    client = get_client()
    wait = client.jobs.run_now(job_id=_require_job_id())
    return wait.run_id


def get_run_status(run_id: int) -> dict[str, Any]:
    """Fetch one run's current status.

    Returns {"life_cycle_state": str, "result_state": str | None,
    "state_message": str | None}. life_cycle_state is one of PENDING,
    QUEUED, RUNNING, TERMINATING, TERMINATED, SKIPPED, INTERNAL_ERROR,
    BLOCKED, WAITING_FOR_RETRY. result_state (e.g. SUCCESS, FAILED) is only
    meaningful once life_cycle_state == "TERMINATED".
    """

    client = get_client()
    run = client.jobs.get_run(run_id)
    state = run.state

    return {
        "life_cycle_state": state.life_cycle_state.value if state and state.life_cycle_state else "UNKNOWN",
        "result_state": state.result_state.value if state and state.result_state else None,
        "state_message": state.state_message if state else None,
    }

if __name__ == "__main__":
    trigger_job()