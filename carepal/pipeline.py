"""note (text or audio) -> verified DailyReport."""
from __future__ import annotations

import json

from pydantic import ValidationError

from . import config
from .grounding import verify
from .llm import LLM, LLMError
from .prompts import extract_system, extract_user
from .rules import add_rule_flags
from .schema import DailyReport, Extraction


def _parse_extraction(raw: str) -> Extraction:
    raw = raw.strip()
    if raw.startswith("```"):  # tolerate fenced JSON
        raw = raw.strip("`")
        raw = raw[raw.find("{") :]
    data = json.loads(raw)
    if isinstance(data, list) and data:
        data = data[0]
    return Extraction.model_validate(data)


def transcribe_audio(llm: LLM, audio: bytes, mime_type: str) -> str:
    if not audio:
        raise ValueError("The recording is empty.")
    if len(audio) > config.MAX_AUDIO_MB * 1024 * 1024:
        raise ValueError(f"Recording is larger than {config.MAX_AUDIO_MB} MB. Please record a shorter note.")
    text = llm.transcribe(audio, mime_type).strip()
    if not text:
        raise ValueError("Could not hear any speech in the recording. Please try again closer to the microphone.")
    return text


def generate_report(note: str, llm: LLM, summary_language: str = "English", date: str | None = None) -> DailyReport:
    note = (note or "").strip()
    if len(note) < 10:
        raise ValueError("The note is too short to make a report. Please write or record a little more.")
    if len(note) > config.MAX_NOTE_CHARS:
        raise ValueError(f"The note is longer than {config.MAX_NOTE_CHARS} characters. Please shorten it or split it by day.")

    lang_desc = config.SUMMARY_LANGUAGES.get(summary_language, summary_language)
    system = extract_system(lang_desc)
    user = extract_user(note)

    extraction: Extraction | None = None
    error = ""
    for attempt in range(2):  # one retry with the validation error fed back
        prompt = user if attempt == 0 else f"{user}\n\nYour previous answer was invalid ({error}). Return valid JSON only."
        raw = llm.generate_json(system, prompt)
        try:
            extraction = _parse_extraction(raw)
            break
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            error = str(e)[:300]
    if extraction is None:
        raise LLMError("The model did not return a usable report. Please try again.")

    report = verify(note, extraction, summary_language, date)
    add_rule_flags(report.extraction)  # after verification: rules only see verified vitals
    return report
