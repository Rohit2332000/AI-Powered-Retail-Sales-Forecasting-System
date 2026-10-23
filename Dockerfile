FROM python:3.12-slim

# uv is a much faster drop-in replacement for pip -- copy the prebuilt
# binary from Astral's official image instead of installing it via pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]