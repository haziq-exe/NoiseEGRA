"""Shared Azure OpenAI helpers for LLM-judge scripts."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AzureOpenAI, RateLimitError


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dotenv_file() -> None:
    load_dotenv(dotenv_path=repo_root() / ".env", override=False)


def require_azure_env() -> tuple[str, str, str, str]:
    load_dotenv_file()
    api_key = os.getenv("AZURE_KEY")
    endpoint = os.getenv("AZURE_ENDPOINT")
    deployment = os.getenv("AZURE_DEPLOYMENT")
    api_version = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

    missing = [
        name
        for name, val in [
            ("AZURE_KEY", api_key),
            ("AZURE_ENDPOINT", endpoint),
            ("AZURE_DEPLOYMENT", deployment),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            "Missing required Azure OpenAI environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in your values."
        )

    return api_key, endpoint, deployment, api_version  # type: ignore[return-value]


def create_azure_client() -> tuple[AzureOpenAI, str]:
    api_key, endpoint, deployment, api_version = require_azure_env()
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    return client, deployment


def chat_with_retry(
    client: AzureOpenAI,
    *,
    model: str,
    messages: Sequence[dict[str, Any]],
    max_retries: int = 5,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(model=model, messages=list(messages))
            return resp.choices[0].message.content or ""
        except RateLimitError as exc:
            last_exc = exc
        except APIConnectionError as exc:
            last_exc = exc
        except APIStatusError as exc:
            if exc.status_code is not None and exc.status_code >= 500:
                last_exc = exc
            else:
                raise
        if attempt < max_retries - 1:
            time.sleep(2**attempt)
    assert last_exc is not None
    raise last_exc
