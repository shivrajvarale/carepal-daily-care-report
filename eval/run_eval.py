"""Evaluate the pipeline on the synthetic notes.

Usage:
    python -m eval.run_eval                  # calls the model (needs GEMINI_API_KEY), writes eval/results.json + EVALUATION.md
    python -m eval.run_eval --rescore        # re-score saved eval/results.json without any API calls
    python -m eval.run_eval --limit 5 --lang Hindi

Two kinds of "invented content" are measured:
  A. caught by the verifier  - quotes/numbers that do not exist in the note (removed before the user sees them)
  B. NOT catchable by string checks - a field filled in although the gold label says the note never mentions it
     (e.g. "no fever" -> temperature). Only gold labels can find these, so this is the number that matters.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from carepal.grounding import numbers_in
from carepal.llm import get_llm
from carepal.pipeline import generate_report
from carepal.schema import DailyReport, Vitals

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "data" / "synthetic_notes.json"
RESULTS = ROOT / "eval" / "results.json"
OUT_MD = ROOT / "EVALUATION.md"
VITALS = list(Vitals.model_fields)


def run_model(notes, language, sleep, limit):
    llm = get_llm()
    results = []
    for n in notes[:limit]:
        t0 = time.time()
        try:
            rep = generate_report(n["note"], llm, language)
            results.append({"id": n["id"], "report": json.loads(rep.model_dump_json()), "error": None, "seconds": round(time.time() - t0, 1)})
        except Exception as e:  # keep going; failures are part of the evaluation
            results.append({"id": n["id"], "report": None, "error": str(e), "seconds": round(time.time() - t0, 1)})
        print(f"{n['id']}: {'ERROR ' + results[-1]['error'] if results[-1]['error'] else 'ok'}")
        time.sleep(sleep)
    return results


def _pct(a, b):
    return f"{100 * a / b:.0f}% ({a}/{b})" if b else "n/a"


def score(notes, results, language):
    by_id = {n["id"]: n for n in notes}
    c = Counter()
    issues: list[str] = []

    for r in results:
        n = by_id[r["id"]]
        g = n["gold"]
        c["notes"] += 1
        if r["error"] or not r["report"]:
            c["failed_notes"] += 1
            issues.append(f"- **{r['id']}** failed: {r['error']}")
            continue
        rep = DailyReport.model_validate(r["report"])
        ex = rep.extraction

        # ---- A. verifier catches
        c["rejected_items"] += len(rep.rejected)
        c["rejected_summary_lines"] += sum(1 for x in rep.rejected if x.where.startswith("summary"))
        c["summary_lines_kept"] += len(ex.summary)
        c["summary_lines_total"] += len(ex.summary) + sum(1 for x in rep.rejected if x.where.startswith("summary"))
        if 4 <= len(ex.summary) <= 6:
            c["summary_len_ok"] += 1
        for x in rep.rejected:
            issues.append(f"- **{r['id']}** verifier removed `{x.where}` ({x.reason}): {x.content[:90]}")

        # ---- vitals
        for name in VITALS:
            got = getattr(ex.vitals, name)
            want = g["vitals"].get(name)
            if want:
                c["vital_gold"] += 1
                if got is None:
                    c["vital_missed"] += 1
                    issues.append(f"- **{r['id']}** missed vital `{name}` (gold {want})")
                else:
                    c["vital_extracted_for_gold"] += 1
                    gotn = sorted(numbers_in(got.value))
                    if any(gotn == sorted(alt) for alt in want):
                        c["vital_correct"] += 1
                    else:
                        c["vital_wrong"] += 1
                        issues.append(f"- **{r['id']}** wrong vital `{name}`: got `{got.value}`, gold {want}")
            else:
                c["vital_absent_gold"] += 1
                if got is not None:
                    c["vital_fabricated"] += 1
                    issues.append(f"- **{r['id']}** FABRICATED vital `{name}` = `{got.value}` (note does not give it)")

        # ---- medicines
        gm = g["medicines"]
        if gm is not None:
            matched_gold = set()
            for m in ex.medicines:
                hay = f"{m.name} {m.quote}".casefold()
                idx = next((i for i, gmed in enumerate(gm) if any(a.casefold() in hay for a in gmed["names"])), None)
                if idx is None:
                    c["med_unexpected"] += 1
                    issues.append(f"- **{r['id']}** medicine not in gold: `{m.name}` ({m.status}) quote: {m.quote[:60]}")
                else:
                    matched_gold.add(idx)
                    want = gm[idx]["status"]
                    ok = m.status == want or (want == "missed" and m.status in ("missed", "refused"))
                    c["med_status_correct" if ok else "med_status_wrong"] += 1
                    if not ok:
                        issues.append(f"- **{r['id']}** medicine `{m.name}` status `{m.status}`, gold `{want}`")
            c["med_gold"] += len(gm)
            c["med_found"] += len(matched_gold)
            for i, gmed in enumerate(gm):
                if i not in matched_gold:
                    issues.append(f"- **{r['id']}** missed medicine {gmed['names'][0]}")

        # ---- presence of sections
        for sect, got in (("meals", bool(ex.meals)), ("activities", bool(ex.activities)), ("mood", ex.mood is not None)):
            want = g[sect]
            if want is None:
                continue
            c[f"{sect}_graded"] += 1
            if got == want:
                c[f"{sect}_ok"] += 1
            elif got and not want:
                c["section_invented"] += 1
                issues.append(f"- **{r['id']}** INVENTED `{sect}` (note does not mention it)")
            else:
                c["section_missed"] += 1
                issues.append(f"- **{r['id']}** missed `{sect}`")

        # ---- concerns
        flagged = any(cn.severity in ("urgent", "attention") for cn in ex.concerns)
        if g["concern"] == "serious":
            c["serious_notes"] += 1
            if flagged:
                c["serious_flagged"] += 1
            else:
                issues.append(f"- **{r['id']}** MISSED serious concern ({n['tag']})")
        elif g["concern"] == "none":
            c["none_notes"] += 1
            if flagged:
                c["false_alarm"] += 1
                issues.append(f"- **{r['id']}** false alarm: {[cn.issue for cn in ex.concerns]}")
        c["rule_flags"] += sum(1 for cn in ex.concerns if cn.source == "rule")

    return c, issues


def render(c: Counter, issues, language, model):
    done = c["notes"] - c["failed_notes"]
    v_found = c["vital_extracted_for_gold"]
    rows = [
        ("Notes processed without error", _pct(done, c["notes"])),
        ("Vitals: recall (gold vitals found)", _pct(v_found, c["vital_gold"])),
        ("Vitals: exact value accuracy (of those found)", _pct(c["vital_correct"], v_found)),
        ("Vitals: **fabricated** (filled although the note does not give it)", _pct(c["vital_fabricated"], c["vital_absent_gold"])),
        ("Medicines: recall", _pct(c["med_found"], c["med_gold"])),
        ("Medicines: status accuracy (given/refused/missed)", _pct(c["med_status_correct"], c["med_status_correct"] + c["med_status_wrong"])),
        ("Medicines: not in gold (possible invention)", str(c["med_unexpected"])),
        ("Meals presence accuracy", _pct(c["meals_ok"], c["meals_graded"])),
        ("Activities presence accuracy", _pct(c["activities_ok"], c["activities_graded"])),
        ("Mood presence accuracy", _pct(c["mood_ok"], c["mood_graded"])),
        ("Sections **invented** (gold says not mentioned)", str(c["section_invented"])),
        ("Serious concerns flagged (recall)", _pct(c["serious_flagged"], c["serious_notes"])),
        ("False alarms on routine notes", _pct(c["false_alarm"], c["none_notes"])),
        ("Items removed by the verifier (quote/number not in note)", str(c["rejected_items"])),
        ("Summary lines removed by the verifier", _pct(c["rejected_summary_lines"], c["summary_lines_total"])),
        ("Summaries of 4-6 lines", _pct(c["summary_len_ok"], done)),
    ]
    md = [f"# Evaluation\n", f"Model: `{model}` | summary language: {language} | {c['notes']} synthetic notes "
          "(12 English, 10 Hinglish, 6 Hindi) with hand-written gold labels in `data/synthetic_notes.json`.\n",
          "_Auto-generated by `python -m eval.run_eval`. Re-run it to refresh the numbers._\n",
          "## Results\n", "| Metric | Result |", "|---|---|"]
    md += [f"| {a} | {b} |" for a, b in rows]
    md += ["\n## How to read the two kinds of invented content\n",
           "- **Caught by the verifier**: the model wrote a quote or number that is not in the note. The app removes it and shows it under *Removed by checker*.",
           "- **Fabricated / invented (rows in bold)**: the field passes the substring check but the gold label says the note never gave it "
           "(for example 'no fever' becoming a temperature). String checks cannot catch these, so they are counted against gold labels.\n",
           "## Every difference from the gold labels\n"]
    md += issues or ["- none"]
    md += ["\n## Analysis\n", "_Write 5-10 lines here after reading the list above: what the model got wrong, why (negation? Hindi digits? "
           "multiple readings?), what the verifier caught versus missed, and what you changed because of it._\n"]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--lang", default="English")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between calls (free-tier rate limits)")
    ap.add_argument("--limit", type=int, default=999)
    a = ap.parse_args()

    notes = json.loads(NOTES.read_text(encoding="utf-8"))
    if a.rescore:
        saved = json.loads(RESULTS.read_text(encoding="utf-8"))
        results, language, model = saved["results"], saved["language"], saved["model"]
    else:
        from carepal import config

        results = run_model(notes, a.lang, a.sleep, a.limit)
        language, model = a.lang, config.GEMINI_MODEL
        RESULTS.write_text(json.dumps({"language": language, "model": model, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    c, issues = score(notes, results, language)
    OUT_MD.write_text(render(c, issues, language, model), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
