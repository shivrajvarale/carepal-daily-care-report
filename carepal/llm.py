
"""Thin model wrapper. Only this file knows which provider is used.

Swap the provider by implementing the two methods of LLM.
"""
from __future__ import annotations

import os
import time
from typing import Protocol

from . import config


class LLMError(RuntimeError):
    """Raised for any provider failure, with a message safe to show a user."""


class LLM(Protocol):
    def generate_json(self, system: str, user: str) -> str: ...

    def transcribe(self, audio: bytes, mime_type: str) -> str: ...


TRANSCRIBE_PROMPT = (
    "Transcribe this voice note exactly as spoken. The speaker may use English, Hindi or Hinglish. "
    "Keep each word in the language and script it was spoken in (write Hinglish in Roman letters, Hindi in Devanagari "
    "if clearly Hindi). Do not translate, summarise, correct or add anything. Write numbers as digits. "
    "If the audio is silent or unintelligible, return an empty string. Output only the transcript."
)


class GeminiLLM:
    def __init__(self, api_key: str, model: str | None = None):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise LLMError(
                "google-genai is not installed. Run: pip install -r requirements.txt"
            ) from e

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model = model or config.GEMINI_MODEL

    def _call(self, contents, system: str | None, json_mode: bool) -> str:

        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json" if json_mode else None,
        )

        last: Exception | None = None

        for attempt in range(3):
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=cfg,
                )

                text = (resp.text or "").strip()

                if not text and json_mode:
                    raise LLMError("The model returned an empty answer.")

                return text

            except LLMError:
                raise

            except Exception as e:
                last = e
                msg = str(e)

                transient = any(
                    code in msg
                    for code in (
                        "429",
                        "500",
                        "503",
                        "504",
                        "UNAVAILABLE",
                        "RESOURCE_EXHAUSTED",
                    )
                )

                if not transient:
                    break

                if attempt < 2:
                    time.sleep(2 * (attempt + 1))

        raise LLMError(
            f"Model call failed: {_short(last)}"
        ) from last

    def generate_json(self, system: str, user: str) -> str:
        return self._call(user, system, json_mode=True)

    def transcribe(self, audio: bytes, mime_type: str) -> str:
        part = self._types.Part.from_bytes(
            data=audio,
            mime_type=mime_type,
        )

        return self._call(
            [TRANSCRIBE_PROMPT, part],
            None,
            json_mode=False,
        )


def _short(e: Exception | None) -> str:
    s = str(e) if e else "unknown error"

    if "API key" in s or "API_KEY" in s or "403" in s or "401" in s:
        return "the API key was rejected - check GEMINI_API_KEY"

    return s[:300]


def get_llm() -> LLM:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not key:
        raise LLMError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add a free key from aistudio.google.com."
        )

    return GeminiLLM(key)
