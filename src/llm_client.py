"""Provider-agnostic LLM client.

Both **Groq** (`https://api.groq.com/openai/v1`) and **DeepSeek**
(`https://api.deepseek.com/v1`) expose an OpenAI-compatible chat-completions
endpoint, so we just swap the `base_url`, `api_key`, and `model` from the
.env file and use the official `openai` SDK.

Public surface:
    - `LLMClient.from_env()`        -> picks provider from `LLM_PROVIDER`
    - `client.complete_json(...)`   -> guarantees a parsed dict, with retries
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

log = logging.getLogger(__name__)


class LLMConfigError(RuntimeError):
    """Bad / missing LLM configuration in the environment."""


class LLMResponseError(RuntimeError):
    """The LLM returned something we cannot use (timeout, bad JSON, etc.)."""


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_blob(text: str) -> str:
    """Pull the first JSON object/array out of an LLM response."""
    if not text:
        raise LLMResponseError("Empty response from LLM.")
    fenced = _JSON_FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    # Otherwise grab the substring from the first { or [ to the matching close.
    first_obj = text.find("{")
    first_arr = text.find("[")
    candidates = [c for c in (first_obj, first_arr) if c != -1]
    if not candidates:
        raise LLMResponseError(f"No JSON found in response: {text[:200]!r}")
    start = min(candidates)
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise LLMResponseError(f"Unbalanced JSON in response: {text[:200]!r}")


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: float
    max_retries: int


class LLMClient:
    """Thin OpenAI-compatible client with JSON-mode + retry semantics."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Build a client from environment variables (default at startup)."""
        load_dotenv(override=False)
        provider = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
        model_override = (
            os.getenv("GROQ_MODEL") if provider == "groq"
            else os.getenv("DEEPSEEK_MODEL")
        )
        return cls.from_choice(provider, model_override or None)

    @classmethod
    def from_choice(
        cls,
        provider: str,
        model: Optional[str] = None,
    ) -> "LLMClient":
        """Build a client for an explicit provider+model selection.

        Used by the GUI/CLI model picker. Falls back to a sensible default
        model per provider when ``model`` is None.
        """
        load_dotenv(override=False)
        provider = (provider or "groq").strip().lower()

        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            chosen = model or os.getenv("GROQ_MODEL") or "meta-llama/llama-4-scout-17b-16e-instruct"
        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            chosen = model or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash"
        else:
            raise LLMConfigError(
                f"Unsupported provider={provider!r}. Use 'groq' or 'deepseek'."
            )

        if not api_key or api_key.startswith("your_"):
            raise LLMConfigError(
                f"Missing API key for provider={provider}. "
                f"Set the relevant *_API_KEY in your .env file."
            )

        return cls(
            LLMConfig(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=chosen,
                timeout=float(os.getenv("LLM_TIMEOUT", "45")),
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            )
        )

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: Optional[str] = None,
        *,
        temperature: float = 0.2,
        force_json_mode: bool = True,
    ) -> dict | list:
        """Send a chat completion that MUST return valid JSON.

        Retries on timeouts, transient network/rate-limit errors, and
        malformed JSON. The final raised error is always `LLMResponseError`.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                user_content = []
                user_content.append({"type": "text", "text": user_prompt})
                if image_base64:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    })

                kwargs: dict = {
                    "model": self.config.model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                }
                if force_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self._client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                blob = _extract_json_blob(content)
                return json.loads(blob)

            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                last_error = exc
                wait = min(2 ** attempt, 8)
                log.warning(
                    "LLM transient error (attempt %d/%d): %s — retrying in %ss",
                    attempt, self.config.max_retries, exc, wait,
                )
                time.sleep(wait)

            except (json.JSONDecodeError, LLMResponseError) as exc:
                last_error = exc
                log.warning(
                    "LLM returned malformed JSON (attempt %d/%d): %s",
                    attempt, self.config.max_retries, exc,
                )
                # Retry without JSON-mode forced — some models flake on it.
                force_json_mode = False
                time.sleep(1)

            except Exception as exc:
                last_error = exc
                log.warning(
                    "Unexpected LLM error (attempt %d/%d): %s",
                    attempt, self.config.max_retries, exc,
                )
                time.sleep(1)

        raise LLMResponseError(
            f"LLM call failed after {self.config.max_retries} attempts: {last_error}"
        )

    def describe(self) -> str:
        return f"{self.config.provider}:{self.config.model}"
