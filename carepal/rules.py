"""Rule-based safety net on *extracted* vitals.

The model may miss that "SpO2 91" matters. These thresholds are deliberately
simple, are shown to the family as "check with nurse", and are NOT clinical
advice. They only ever add a flag; they never change a reading.
"""
from __future__ import annotations

import re

from .schema import Concern, Evidenced, Extraction

_N = re.compile(r"\d+(?:\.\d+)?")


def _nums(s: str) -> list[float]:
    return [float(x) for x in _N.findall(s)]


def vital_rule_concerns(ex: Extraction) -> list[Concern]:
    out: list[Concern] = []
    v = ex.vitals

    def add(item: Evidenced, issue: str, sev: str) -> None:
        out.append(Concern(issue=issue, severity=sev, quote=item.quote, source="rule"))

    if v.spo2:
        n = _nums(v.spo2.value)
        if n and n[0] < 92:
            add(v.spo2, f"Low oxygen saturation ({v.spo2.value})", "urgent")
        elif n and n[0] < 95:
            add(v.spo2, f"Oxygen saturation a little low ({v.spo2.value})", "attention")
    if v.temperature:
        n = _nums(v.temperature.value)
        val = v.temperature.value.lower()
        if n:
            t = n[0]
            celsius = ("c" in val and "f" not in val) or t < 50
            f = t * 9 / 5 + 32 if celsius else t
            if f >= 100.4:
                add(v.temperature, f"Fever ({v.temperature.value})", "attention")
    if v.blood_pressure:
        n = _nums(v.blood_pressure.value)
        if len(n) >= 2:
            sys_, dia = n[0], n[1]
            if sys_ >= 160 or dia >= 100:
                add(v.blood_pressure, f"High blood pressure ({v.blood_pressure.value})", "attention")
            elif sys_ < 90 or dia < 60:
                add(v.blood_pressure, f"Low blood pressure ({v.blood_pressure.value})", "attention")
    if v.pulse:
        n = _nums(v.pulse.value)
        if n and (n[0] > 110 or n[0] < 50):
            add(v.pulse, f"Pulse out of usual range ({v.pulse.value})", "attention")
    if v.blood_sugar:
        n = _nums(v.blood_sugar.value)
        if n and (n[0] < 70 or n[0] > 250):
            add(v.blood_sugar, f"Blood sugar out of range ({v.blood_sugar.value})", "attention")
    return out


def add_rule_flags(ex: Extraction) -> None:
    """Append rule concerns unless the note-based concerns already cover that quote."""
    existing = {c.quote.strip().casefold() for c in ex.concerns}
    for c in vital_rule_concerns(ex):
        if c.quote.strip().casefold() not in existing:
            ex.concerns.append(c)
