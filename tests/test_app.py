"""UI smoke test: runs app.py headlessly with a fake model (no API key needed)."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

import pytest
import streamlit as st

import carepal.llm as llm_mod


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield

APP = str(Path(__file__).resolve().parent.parent / "app.py")
NOTE = "Aaj subah BP 130/80, pulse 74. Breakfast mein 2 roti khayi. Metformin 500 mg de di. Mood achha tha. SpO2 91%."


class Fake:
    def generate_json(self, system, user):
        if '"lines"' in system:  # weekly prompt
            return json.dumps({"lines": [{"text": "Is hafte 7 din ke notes mile.", "days": [1, 2]}]})
        return json.dumps({
            "vitals": {"blood_pressure": {"value": "130/80 mmHg", "quote": "BP 130/80"},
                       "spo2": {"value": "91 %", "quote": "SpO2 91%"}},
            "meals": [{"meal": "breakfast", "detail": "2 roti", "quote": "Breakfast mein 2 roti khayi"}],
            "mood": {"value": "good", "quote": "Mood achha tha"},
            "summary": [{"text": "Unhone breakfast mein 2 roti khayi.", "evidence": ["Breakfast mein 2 roti khayi"]},
                        {"text": "BP 999/1 tha.", "evidence": ["BP 130/80"]}],
        })

    def transcribe(self, audio, mime_type):
        return NOTE


def test_daily_flow(monkeypatch):
    monkeypatch.setattr(llm_mod, "get_llm", lambda: Fake())
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.text_area(key="note_text").set_value(NOTE).run()
    next(b for b in at.button if b.label == "Create report").click().run()
    assert not at.exception, at.exception
    page = " ".join(m.value for m in at.markdown) + " ".join(s.value for s in at.subheader)
    assert "Needs attention" in page            # SpO2 91 -> rule flag shown first
    assert "Urgent" in page
    assert "2 roti" in page                     # verified summary line shown
    # the invented-number line must be listed as removed, not shown as a summary line
    assert any("Removed by the checker" in e.label for e in at.expander)
    assert not any("BP 999/1 tha." in m.value and "Source" in m.value for m in at.markdown)


def test_short_note_shows_error(monkeypatch):
    monkeypatch.setattr(llm_mod, "get_llm", lambda: Fake())
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="note_text").set_value("hi").run()
    next(b for b in at.button if b.label == "Create report").click().run()
    assert at.error and "too short" in at.error[0].value


def test_missing_key_shows_friendly_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="note_text").set_value(NOTE).run()
    next(b for b in at.button if b.label == "Create report").click().run()
    assert not at.exception
    assert at.error and "GEMINI_API_KEY" in at.error[0].value
