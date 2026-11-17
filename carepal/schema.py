"""Data models for the daily report.

Design rule: every extracted fact carries a ``quote`` -- a verbatim span of
the nurse's note. A fact whose quote cannot be found in the note is dropped
by ``carepal.grounding``. Anything the nurse did not mention stays ``None``
or an empty list; there are no default values that look like data.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Evidenced(BaseModel):
    """A single value plus the exact words of the note it came from."""

    value: str
    quote: str


class Vitals(BaseModel):
    blood_pressure: Optional[Evidenced] = None  # e.g. "130/80 mmHg"
    pulse: Optional[Evidenced] = None  # e.g. "78 bpm"
    temperature: Optional[Evidenced] = None  # e.g. "99.1 F"
    spo2: Optional[Evidenced] = None  # e.g. "97 %"
    blood_sugar: Optional[Evidenced] = None  # e.g. "142 mg/dL (fasting)"
    respiratory_rate: Optional[Evidenced] = None
    weight: Optional[Evidenced] = None


class Meal(BaseModel):
    meal: Literal["breakfast", "lunch", "dinner", "snack", "fluids", "unspecified"]
    detail: str  # what / how much was eaten, in the nurse's terms
    quote: str


class Medicine(BaseModel):
    name: str
    dose: Optional[str] = None
    time: Optional[str] = None
    status: Literal["given", "refused", "missed", "unspecified"]
    quote: str


class Activity(BaseModel):
    activity: str
    quote: str


class Concern(BaseModel):
    issue: str
    severity: Literal["urgent", "attention", "monitor"]
    quote: str
    source: Literal["note", "rule"] = "note"  # "rule" = raised by a vital-sign rule


class SummaryLine(BaseModel):
    text: str  # in the family's preferred language
    evidence: list[str]  # verbatim quotes from the note supporting this line


class Extraction(BaseModel):
    """What the model returns for one note (before verification)."""

    vitals: Vitals = Field(default_factory=Vitals)
    meals: list[Meal] = Field(default_factory=list)
    medicines: list[Medicine] = Field(default_factory=list)
    activities: list[Activity] = Field(default_factory=list)
    mood: Optional[Evidenced] = None
    concerns: list[Concern] = Field(default_factory=list)
    summary: list[SummaryLine] = Field(default_factory=list)


class Rejected(BaseModel):
    where: str  # e.g. "medicines[1]" or "summary[3]"
    reason: str
    content: str


class DailyReport(BaseModel):
    """Verified report. Everything in here is traceable to ``note``."""

    note: str
    language_of_summary: str
    date: Optional[str] = None
    extraction: Extraction
    rejected: list[Rejected] = Field(default_factory=list)  # removed by the verifier
    warnings: list[str] = Field(default_factory=list)  # soft issues, shown to the user
    not_mentioned: list[str] = Field(default_factory=list)  # sections left empty


# ---- weekly -----------------------------------------------------------------

class WeeklyLine(BaseModel):
    text: str
    days: list[int]  # 1-based indexes of the daily reports used


class WeeklySummary(BaseModel):
    language: str
    lines: list[WeeklyLine]
    stats: dict  # computed in code, never by the model
    warnings: list[str] = Field(default_factory=list)
