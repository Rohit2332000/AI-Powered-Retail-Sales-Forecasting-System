
"""Groq LLM client shared by POST /forecast/explain and POST /chat.

Reads credentials from environment variables loaded from the project
root .env file. Never hardcode API keys.
"""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

# Load .env from the backend project root.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

_client: Optional[OpenAI] = None


class LLMConfigError(RuntimeError):
    """Raised when required Groq environment variables are missing."""


def get_client() -> OpenAI:
    """Return a cached Groq client or raise a configuration error."""
    global _client

    if _client is None:
        missing = [
            name
            for name, value in {
                "GROQ_API_KEY": GROQ_API_KEY,
                "GROQ_MODEL": GROQ_MODEL,
            }.items()
            if not value
        ]

        if missing:
            raise LLMConfigError(
                f"Missing environment variable(s): {', '.join(missing)}"
            )

        _client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )

    return _client


def call_chat_json(system_prompt: str, user_message: str) -> str:
    """Generate a JSON-formatted response using Groq."""
    client = get_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Groq returned an empty response.")

    return content


def call_chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    json_mode: bool = False,
) -> ChatCompletionMessage:
    """General-purpose chat completion with optional tools and JSON mode."""
    client = get_client()

    kwargs: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick manual test for the Groq LLM client."
    )
    parser.add_argument(
        "--name",
        default="World",
        help="Name to greet in the test prompt",
    )
    args = parser.parse_args()

    reply = call_chat_json(
        'You are a helpful assistant. Respond in JSON with this shape: '
        '{"message": "..."}.',
        f"Say hello to {args.name} in one short sentence.",
    )
    print(reply)
