"""Verification layer: the model is not trusted, its output is checked.

What is checked (all deterministic, no second LLM call):
1. Every quote must be a verbatim substring of the note (after harmless
   normalisation: case, whitespace, unicode form, Devanagari digits).
2. Vital values must contain the numbers that appear in the quote.
3. Summary lines need >=1 valid quote, and every number written in a summary
   line must exist in the note (catches invented figures, even when the
   text is in another language).
4. Facts that fail are removed and listed in ``rejected`` -- never silently
   kept.
"""
from __future__ import annotations

import re
import unicodedata

from .schema import (
    DailyReport,
    Extraction,
    Rejected,
    Vitals,
)

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_PUNCT_EDGES = " \t\r\n\"'“”‘’.,;:!?-–—()[]"


def normalise(text: str) -> str:
    """Normalise for *comparison only* (never for display)."""
    text = unicodedata.normalize("NFC", text).translate(_DEVANAGARI_DIGITS)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(_PUNCT_EDGES)


def quote_in_note(quote: str, note: str) -> bool:
    q = normalise(quote)
    if len(q) < 2:
        return False
    return q in normalise(note)


_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> set[str]:
    t = unicodedata.normalize("NFC", text).translate(_DEVANAGARI_DIGITS)
    return {n.replace(",", ".") for n in _NUM.findall(t)}


# Words that turn into digits in a summary but may be written as words in the
# note ("ek", "two"). Small whitelist so "one meal" is not flagged as invented.
_SMALL_NUMBER_WORDS = {"1", "2", "3", "4", "5"}


def _check_numbers(summary_text: str, note: str) -> set[str]:
    """Numbers present in the summary text but absent from the note."""
    note_nums = numbers_in(note)
    missing = numbers_in(summary_text) - note_nums
    return {m for m in missing if m not in _SMALL_NUMBER_WORDS}


def _verify_vitals(v: Vitals, note: str, rejected: list[Rejected]) -> Vitals:
    cleaned = Vitals()
    for name in Vitals.model_fields:
        item = getattr(v, name)
        if item is None:
            continue
        where = f"vitals.{name}"
        if not quote_in_note(item.quote, note):
            rejected.append(Rejected(where=where, reason="quote not found in note", content=item.value))
            continue
        # every number in the value must be in the quote -> no invented readings
        extra = numbers_in(item.value) - numbers_in(item.quote)
        if extra:
            rejected.append(
                Rejected(where=where, reason=f"value has numbers not in quote: {sorted(extra)}", content=item.value)
            )
            continue
        setattr(cleaned, name, item)
    return cleaned


def _keep_grounded(items: list, field: str, note: str, rejected: list[Rejected]) -> list:
    kept = []
    for i, it in enumerate(items):
        if quote_in_note(it.quote, note):
            kept.append(it)
        else:
            rejected.append(
                Rejected(where=f"{field}[{i}]", reason="quote not found in note", content=it.model_dump_json())
            )
    return kept


def verify(note: str, ex: Extraction, summary_language: str, date: str | None = None) -> DailyReport:
    rejected: list[Rejected] = []
    warnings: list[str] = []

    out = Extraction()
    out.vitals = _verify_vitals(ex.vitals, note, rejected)
    out.meals = _keep_grounded(ex.meals, "meals", note, rejected)
    out.medicines = _keep_grounded(ex.medicines, "medicines", note, rejected)
    out.activities = _keep_grounded(ex.activities, "activities", note, rejected)
    out.concerns = _keep_grounded(ex.concerns, "concerns", note, rejected)

    if ex.mood is not None:
        if quote_in_note(ex.mood.quote, note):
            out.mood = ex.mood
        else:
            rejected.append(Rejected(where="mood", reason="quote not found in note", content=ex.mood.value))

    # medicine doses must come from the note
    for m in out.medicines:
        if m.dose and numbers_in(m.dose) - numbers_in(m.quote):
            warnings.append(f"Dose '{m.dose}' for {m.name} has a number not found in its quote - please check.")

    # summary
    for i, line in enumerate(ex.summary):
        good = [q for q in line.evidence if quote_in_note(q, note)]
        if not good:
            rejected.append(Rejected(where=f"summary[{i}]", reason="no evidence quote found in note", content=line.text))
            continue
        bad_nums = _check_numbers(line.text, note)
        if bad_nums:
            rejected.append(
                Rejected(where=f"summary[{i}]", reason=f"numbers not in note: {sorted(bad_nums)}", content=line.text)
            )
            continue
        out.summary.append(line.model_copy(update={"evidence": good}))

    if ex.summary and not out.summary:
        warnings.append("Every summary line failed verification; showing the structured report only.")

    not_mentioned = []
    for name in Vitals.model_fields:
        if getattr(out.vitals, name) is None:
            not_mentioned.append(f"vitals.{name}")
    for sect in ("meals", "medicines", "activities"):
        if not getattr(out, sect):
            not_mentioned.append(sect)
    if out.mood is None:
        not_mentioned.append("mood")

    return DailyReport(
        note=note,
        language_of_summary=summary_language,
        date=date,
        extraction=out,
        rejected=rejected,
        warnings=warnings,
        not_mentioned=not_mentioned,
    )
