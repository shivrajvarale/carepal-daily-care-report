# Video script (target 4:00, limit 5:00) - screen recording

**0:00 - 0:20 Intro.** "This turns a nurse's daily note - typed or spoken, English, Hindi or Hinglish - into a structured report and a short, warm summary for the family. Every line links back to the note."

**0:20 - 1:30 Demo (working product first).**
1. Phone-width browser window. Pick sample **n11** (Hinglish fever note) → Create report.
2. Point at: *Needs attention* is on top; the summary lines each show a **Source** quote; "Show the note with sources highlighted".
3. Structured report: vitals not in the note say **Not mentioned** (BP - the nurse wrote "BP check nahi kiya").
4. Switch language to Hindi → regenerate. Then **Voice note**: record 10 seconds in Hinglish → Transcribe → fix a word → create report.

**1:30 - 2:15 Weekly + edge cases.** Weekly tab → Load sample week → Create weekly summary. Show "numbers behind the summary". Open sample **n26** (injection) and **n07** ("no fever") - temperature stays empty.

**2:15 - 3:30 Code walkthrough.**
- `schema.py`: every fact has a `quote`.
- `grounding.py`: quote must be in the note; numbers must be in the note; failures are removed and listed.
- `rules.py`: vital-sign flags in code, labelled not-medical-advice.
- `pipeline.py` / `llm.py`: retry on bad JSON, friendly errors, no secrets (`.env.example`).
- `weekly.py`: stats in Python, model only words them.

**3:30 - 4:15 Decisions + honesty.** Why two-step voice. Why rules in code. Show `EVALUATION.md`: headline numbers, one thing the model got wrong and the limit of string checking. What I'd build next (nurse approval, WhatsApp/FCM, per-family profile).
