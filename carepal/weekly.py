"""Weekly summary from up to seven verified daily reports.

Statistics are computed in code. The model only writes the narrative from
those facts, and every number it writes is checked against them.
"""
from __future__ import annotations

import json
from collections import defaultdict

from . import config
from .grounding import numbers_in
from .llm import LLM, LLMError
from .prompts import weekly_system
from .schema import DailyReport, WeeklyLine, WeeklySummary

_ALLOWED_SMALL = {str(i) for i in range(1, 8)}  # "7 days", "3 days"


def compute_stats(reports: list[DailyReport]) -> dict:
    vitals: dict[str, list[dict]] = defaultdict(list)
    meds: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    concerns: list[dict] = []
    moods: list[dict] = []
    days_without_note_data = []

    for i, r in enumerate(reports, start=1):
        ex = r.extraction
        had_anything = False
        for name in type(ex.vitals).model_fields:
            item = getattr(ex.vitals, name)
            if item:
                vitals[name].append({"day": i, "value": item.value})
                had_anything = True
        for m in ex.medicines:
            meds[m.name.strip().title()][m.status] += 1
            had_anything = True
        for c in ex.concerns:
            concerns.append({"day": i, "issue": c.issue, "severity": c.severity})
            had_anything = True
        if ex.mood:
            moods.append({"day": i, "mood": ex.mood.value})
            had_anything = True
        if ex.meals or ex.activities:
            had_anything = True
        if not had_anything:
            days_without_note_data.append(i)

    return {
        "days_covered": len(reports),
        "vitals_by_day": dict(vitals),
        "medicines": {k: dict(v) for k, v in meds.items()},
        "concerns": concerns,
        "moods": moods,
        "meals_recorded_on_days": [i for i, r in enumerate(reports, 1) if r.extraction.meals],
        "activities_recorded_on_days": [i for i, r in enumerate(reports, 1) if r.extraction.activities],
        "days_with_nothing_recorded": days_without_note_data,
    }


def generate_weekly(reports: list[DailyReport], llm: LLM, summary_language: str = "English") -> WeeklySummary:
    if not 1 <= len(reports) <= 7:
        raise ValueError("Please provide between 1 and 7 daily reports.")
    stats = compute_stats(reports)
    day_facts = [
        {
            "day": i,
            "date": r.date,
            "summary": [s.text for s in r.extraction.summary],
            "mood": r.extraction.mood.value if r.extraction.mood else None,
            "concerns": [c.issue for c in r.extraction.concerns],
        }
        for i, r in enumerate(reports, 1)
    ]
    payload = {"stats": stats, "days": day_facts}
    payload_text = json.dumps(payload, ensure_ascii=False)

    lang_desc = config.SUMMARY_LANGUAGES.get(summary_language, summary_language)
    raw = llm.generate_json(weekly_system(lang_desc), payload_text)
    try:
        data = json.loads(raw.strip().strip("`").removeprefix("json"))
        lines = [WeeklyLine.model_validate(x) for x in data["lines"]]
    except Exception as e:
        raise LLMError("The model did not return a usable weekly summary. Please try again.") from e

    allowed_numbers = numbers_in(payload_text) | _ALLOWED_SMALL
    kept, warnings = [], []
    valid_days = set(range(1, len(reports) + 1))
    for ln in lines:
        days = [d for d in ln.days if d in valid_days]
        bad = numbers_in(ln.text) - allowed_numbers
        if not days:
            warnings.append(f"Removed a line that cited no valid day: {ln.text[:80]}")
        elif bad:
            warnings.append(f"Removed a line with numbers not in the daily reports {sorted(bad)}: {ln.text[:80]}")
        else:
            kept.append(WeeklyLine(text=ln.text, days=sorted(set(days))))
    return WeeklySummary(language=summary_language, lines=kept, stats=stats, warnings=warnings)
