"""Streamlit demo: nurse note (text / voice) -> structured report + family summary.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import streamlit as st

from carepal import config
from carepal.llm import LLMError, get_llm
from carepal.pipeline import generate_report, transcribe_audio
from carepal.schema import DailyReport, Vitals
from carepal.weekly import generate_weekly

DATA = Path(__file__).parent / "data"
VITAL_LABELS = {
    "blood_pressure": "Blood pressure",
    "pulse": "Pulse",
    "temperature": "Temperature",
    "spo2": "SpO2",
    "blood_sugar": "Blood sugar",
    "respiratory_rate": "Respiratory rate",
    "weight": "Weight",
}
SEV_STYLE = {"urgent": ("🔴", "Urgent"), "attention": ("🟠", "Needs attention"), "monitor": ("🟡", "Keep an eye on")}

st.set_page_config(page_title="Daily Care Report", page_icon="🩺", layout="centered")
st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; max-width: 760px;}
.cp-card {border-radius: 12px; padding: 0.8rem 1rem; margin: 0.4rem 0; border: 1px solid rgba(128,128,128,.3);}
.cp-flag-urgent {border-left: 6px solid #d62828;}
.cp-flag-attention {border-left: 6px solid #f77f00;}
.cp-flag-monitor {border-left: 6px solid #e9c46a;}
.cp-src {font-size: .8rem; opacity: .75;}
mark {background: #ffe66d; color: #111; padding: 0 2px; border-radius: 3px;}
@media (max-width: 640px) {.block-container {padding-left: .8rem; padding-right: .8rem;}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _llm():
    return get_llm()


def _highlight(note: str, quotes: list[str]) -> str:
    """Return the note as HTML with every evidence quote wrapped in <mark>."""
    spans: list[tuple[int, int]] = []
    for q in quotes:
        toks = q.split()
        if not toks:
            continue
        pat = r"\s+".join(re.escape(t) for t in toks)
        m = re.search(pat, note, flags=re.IGNORECASE)
        if m:
            spans.append(m.span())
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, pos = [], 0
    for s, e in merged:
        out += [html.escape(note[pos:s]), "<mark>", html.escape(note[s:e]), "</mark>"]
        pos = e
    out.append(html.escape(note[pos:]))
    return "".join(out).replace("\n", "<br>")


def _all_quotes(r: DailyReport) -> list[str]:
    ex = r.extraction
    q = [s_q for s in ex.summary for s_q in s.evidence]
    return q


def _family_message(r: DailyReport) -> str:
    return "\n".join(s.text for s in r.extraction.summary)


def render_report(r: DailyReport, key: str):
    ex = r.extraction

    # 1) concerns first -- the family should see these before anything else
    st.subheader("Needs attention")
    if ex.concerns:
        order = {"urgent": 0, "attention": 1, "monitor": 2}
        for c in sorted(ex.concerns, key=lambda c: order[c.severity]):
            icon, label = SEV_STYLE[c.severity]
            by = " · flagged by a vital-sign rule" if c.source == "rule" else ""
            st.markdown(
                f'<div class="cp-card cp-flag-{c.severity}"><b>{icon} {label}:</b> {html.escape(c.issue)}'
                f'<div class="cp-src">From the note: “{html.escape(c.quote)}”{by}</div></div>',
                unsafe_allow_html=True,
            )
        if any(c.source == "rule" for c in ex.concerns):
            st.caption("Rule-based flags are simple range checks, not medical advice. Please speak to the nurse or doctor.")
    else:
        st.success("No concerns were mentioned in this note.")

    # 2) family summary with the source of every line
    st.subheader("Summary for the family")
    if ex.summary:
        for i, s in enumerate(ex.summary, 1):
            src = " · ".join(f"“{html.escape(q)}”" for q in s.evidence)
            st.markdown(
                f"<div style='margin-bottom:.7rem'><b>{i}.</b> {html.escape(s.text)}"
                f"<div class='cp-src'>Source: {src}</div></div>",
                unsafe_allow_html=True,
            )
        st.text_area("Copy to send on WhatsApp", _family_message(r), height=140, key=f"msg_{key}")
        with st.expander("Show the note with sources highlighted"):
            st.markdown(_highlight(r.note, _all_quotes(r)), unsafe_allow_html=True)
    else:
        st.info("No verified summary lines could be produced for this note.")

    # 3) structured report
    st.subheader("Structured report")
    rows = []
    for name, label in VITAL_LABELS.items():
        item = getattr(ex.vitals, name)
        rows.append({"Vital": label, "Value": item.value if item else "—", "From the note": item.quote if item else "not mentioned"})
    st.dataframe(rows, hide_index=True, width="stretch")

    def section(title, items, cols):
        st.markdown(f"**{title}**")
        if items:
            st.dataframe([cols(i) for i in items], hide_index=True, width="stretch")
        else:
            st.caption("Not mentioned")

    section("Meals", ex.meals, lambda m: {"Meal": m.meal, "What": m.detail, "From the note": m.quote})
    section(
        "Medicines",
        ex.medicines,
        lambda m: {"Medicine": m.name, "Dose": m.dose or "—", "Time": m.time or "—", "Status": m.status, "From the note": m.quote},
    )
    section("Activities", ex.activities, lambda a: {"Activity": a.activity, "From the note": a.quote})
    st.markdown("**Mood**")
    if ex.mood:
        st.write(f"{ex.mood.value}  \n<span class='cp-src'>“{html.escape(ex.mood.quote)}”</span>", unsafe_allow_html=True)
    else:
        st.caption("Not mentioned")

    for w in r.warnings:
        st.warning(w)
    if r.rejected:
        with st.expander(f"Removed by the checker ({len(r.rejected)})"):
            st.caption("The model wrote these, but they could not be found in the note, so they were removed.")
            for x in r.rejected:
                st.markdown(f"- `{x.where}` — {x.reason}: {x.content[:200]}")

    st.download_button(
        "Download report (JSON)",
        r.model_dump_json(indent=2, exclude={"note"}),
        file_name="daily_report.json",
        mime="application/json",
        key=f"dl_{key}",
    )


@st.cache_data(show_spinner=False)
def _samples() -> list[dict]:
    return json.loads((DATA / "synthetic_notes.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ UI
st.title("🩺 Daily Care Report")
st.caption("Turn a nurse's daily note (typed or spoken, English / Hindi / Hinglish) into a clear update for the family. Demo with synthetic data only.")

tab_daily, tab_week, tab_about = st.tabs(["Daily report", "Weekly summary", "How it works"])

with tab_daily:
    lang = st.selectbox("Family's preferred language", list(config.SUMMARY_LANGUAGES), index=0, key="lang")

    samples = _samples()
    labels = ["— choose a sample note —"] + [f"{n['id']} · {n['language']} · {n['tag'] or 'general'}" for n in samples]
    pick = st.selectbox("Try a sample note", labels)
    if pick != labels[0] and st.session_state.get("_picked") != pick:
        st.session_state["_picked"] = pick
        st.session_state["note_text"] = samples[labels.index(pick) - 1]["note"]

    with st.expander("🎤 Voice note", expanded=False):
        audio = st.audio_input("Record the note")
        up = st.file_uploader("…or upload a recording", type=["wav", "mp3", "m4a", "ogg", "webm", "flac"])
        src = audio or up
        if src is not None and st.button("Transcribe voice note"):
            try:
                mime = getattr(src, "type", None) or "audio/wav"
                with st.spinner("Listening…"):
                    st.session_state["note_text"] = transcribe_audio(_llm(), src.getvalue(), mime)
                st.success("Transcribed. Check the text below, fix anything, then create the report.")
            except (ValueError, LLMError) as e:
                st.error(str(e))

    note = st.text_area("Nurse's note", key="note_text", height=170, placeholder="e.g. Aaj subah BP 130/80, pulse 74. Breakfast mein 2 roti khayi…")
    st.caption(f"{len(note)}/{config.MAX_NOTE_CHARS} characters")

    if st.button("Create report", type="primary", width="stretch"):
        try:
            with st.spinner("Reading the note and checking every line against it…"):
                st.session_state["report"] = generate_report(note, _llm(), lang)
        except (ValueError, LLMError) as e:
            st.session_state.pop("report", None)
            st.error(str(e))

    rep = st.session_state.get("report")
    if rep:
        st.divider()
        render_report(rep, "daily")
        if st.button("➕ Add this report to the weekly summary"):
            week = st.session_state.setdefault("week", [])
            if len(week) >= 7:
                st.warning("A week holds at most 7 daily reports.")
            else:
                week.append(rep)
                st.success(f"Added as day {len(week)}.")

with tab_week:
    st.write("Combine up to seven daily reports into one weekly update. Numbers are counted by code; the model only writes the wording, and every line is checked.")
    wlang = st.selectbox("Summary language", list(config.SUMMARY_LANGUAGES), index=1, key="wlang")
    week: list[DailyReport] = st.session_state.setdefault("week", [])

    c1, c2 = st.columns(2)
    if c1.button("Load sample week (7 notes)", width="stretch"):
        sample = json.loads((DATA / "sample_week.json").read_text(encoding="utf-8"))
        try:
            week.clear()
            bar = st.progress(0.0, text="Creating daily reports…")
            for i, d in enumerate(sample["notes"], 1):
                week.append(generate_report(d["note"], _llm(), wlang, d.get("date")))
                bar.progress(i / 7, text=f"Day {i} of 7")
        except (ValueError, LLMError) as e:
            st.error(str(e))
    if c2.button("Clear week", width="stretch"):
        week.clear()
        st.session_state.pop("weekly", None)

    if week:
        st.markdown(f"**{len(week)} daily report(s) in this week**")
        for i, r in enumerate(week, 1):
            n_flags = sum(1 for c in r.extraction.concerns if c.severity in ("urgent", "attention"))
            st.caption(f"Day {i}{' · ' + r.date if r.date else ''} — {len(r.extraction.summary)} summary lines" + (f" · ⚠ {n_flags} concern(s)" if n_flags else ""))
        if st.button("Create weekly summary", type="primary", width="stretch"):
            try:
                with st.spinner("Writing the weekly update…"):
                    st.session_state["weekly"] = generate_weekly(week, _llm(), wlang)
            except (ValueError, LLMError) as e:
                st.error(str(e))
    else:
        st.info("No daily reports yet. Add them from the Daily report tab or load the sample week.")

    w = st.session_state.get("weekly")
    if w:
        st.divider()
        st.subheader("This week")
        for ln in w.lines:
            st.markdown(f"- {ln.text}  \n<span class='cp-src'>based on day(s) {', '.join(map(str, ln.days))}</span>", unsafe_allow_html=True)
        for msg in w.warnings:
            st.warning(msg)
        st.text_area("Copy to send on WhatsApp", "\n".join(ln.text for ln in w.lines), height=160, key="wk_msg")
        with st.expander("Numbers behind the summary (computed by code)"):
            st.json(w.stats)

with tab_about:
    st.markdown(
        """
**How "nothing is made up" is enforced**

1. The model must copy the exact words from the note (a *quote*) next to every fact and every summary line.
2. Code then looks for each quote in the note. If it is not there, the item is removed and listed under *Removed by the checker*.
3. A number written in a summary or a vital must also exist in the note.
4. Anything the nurse did not mention stays empty and shows **Not mentioned**.
5. Simple vital-sign rules (for example SpO2 below 92%) add a flag even if the model missed it.

Limits: this is a demo on synthetic notes. It is not medical advice and does not replace the nurse or the doctor.
"""
    )
