import json
from collections import Counter

from eval import run_eval


class FakeReport:
    def model_dump_json(self):
        return json.dumps({"id": "report"})


def test_run_model_resumes_successes_and_checkpoints_retries(monkeypatch):
    class FakeLLM:
        pass

    calls = []
    checkpoints = []
    monkeypatch.setattr(run_eval, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(
        run_eval,
        "generate_report",
        lambda note, llm, language: calls.append(note) or FakeReport(),
    )
    monkeypatch.setattr(run_eval.time, "sleep", lambda _: None)

    notes = [{"id": f"n{i}", "note": f"note {i}"} for i in range(1, 4)]
    saved = [
        {"id": "n1", "report": {"id": "saved"}, "error": None, "seconds": 1.0},
        {"id": "n2", "report": None, "error": "quota", "seconds": 2.0},
    ]

    results = run_eval.run_model(
        notes,
        "English",
        sleep=0,
        limit=3,
        existing_results=saved,
        on_update=checkpoints.append,
    )

    assert calls == ["note 2", "note 3"]
    assert [result["id"] for result in results] == ["n1", "n2", "n3"]
    assert results[0]["report"] == {"id": "saved"}
    assert results[1]["error"] is None
    assert checkpoints[-1] == results


def test_render_labels_partial_results_and_reports_success_latency():
    report = run_eval.render(
        Counter(notes=2, failed_notes=1, section_invented=1, rejected_items=0),
        [],
        "English",
        "gemini-3.8-flash",
        [
            {"id": "n1", "report": {"summary": []}, "error": None, "seconds": 12.3},
            {"id": "n2", "report": None, "error": "quota", "seconds": 30.0},
        ],
    )

    assert "Run status:** 1/2 notes succeeded; 1 failed." in report
    assert "12.3 s median (n=1; range 12.3-12.3 s)" in report
    assert "only to successful notes" in report
