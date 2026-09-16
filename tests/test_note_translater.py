import json
import os
import pytest
from conftest import PAGE_OK, SUMMARY_OK

import note_translater as nt


class RateLimitError(Exception):
    pass


def _updates_recorder():
    calls = []

    def on_update(group_name, **fields):
        calls.append((group_name, fields))
    return calls, on_update


def _run(group_folder, script, **kw):
    from conftest import ScriptedApi
    api = ScriptedApi(script)
    calls, on_update = _updates_recorder()
    result = nt.process_group(
        "n1", on_update=on_update, input_dir=str(group_folder.parent),
        call_api_fn=api, run_id="r1", sleep_fn=lambda s: None, **kw)
    return result, api, calls


def test_two_pages_all_ok(group_folder):
    result, api, calls = _run(group_folder, [json.dumps(PAGE_OK), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])
    assert result.status == "done"
    assert [p.status for p in result.pages] == ["done", "done"]
    assert result.summary["title"] == "Two page note"
    assert result.summary_attempts == 1
    assert result.pages[0].meta.kind == "page" and result.summary_meta.kind == "summary"
    assert result.pages[0].meta.prompt_hash != result.summary_meta.prompt_hash
    assert len(api.calls) == 3
    # first page call carries prompt, OCR text and image in that order
    blocks = api.calls[0][0]["content"]
    assert blocks[0]["type"] == "text" and "uncertain" in blocks[0]["text"]
    assert blocks[1] == {"type": "text", "text": "ocr text one"}
    assert blocks[2]["type"] == "image"


def test_on_update_sequence(group_folder):
    _, _, calls = _run(group_folder, [json.dumps(PAGE_OK), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])
    stages = [f.get("stage") for _, f in calls if "stage" in f]
    assert stages == ["transcribing", "summarising", "complete"]
    done_counts = [f["pages_done"] for _, f in calls if "pages_done" in f]
    assert done_counts == [0, 1, 2]
    assert calls[-1][1]["status"] == "done"


def test_failed_page_keeps_sibling_and_fails_group(group_folder):
    result, api, calls = _run(group_folder, [json.dumps(PAGE_OK), "{a", "{b", "{c", json.dumps(SUMMARY_OK)])
    assert result.status == "failed"
    assert result.pages[0].status == "done" and result.pages[0].data["title"] == "Grocery list"
    assert result.pages[1].status == "failed" and result.pages[1].attempts == 3
    assert "page2.jpg" in result.errors[0] and "after 3 attempts" in result.errors[0]
    assert calls[-1][1]["status"] == "failed" and "page2.jpg" in calls[-1][1]["error"]
    # summary still attempted so the retry surface is complete
    assert result.summary is not None


def test_summary_failure_fails_group(group_folder):
    result, *_ = _run(group_folder, [json.dumps(PAGE_OK), json.dumps(PAGE_OK), RateLimitError("x"), RateLimitError("y"), RateLimitError("z")])
    assert result.status == "failed" and result.summary is None
    assert result.summary_attempts == 3 and "summary" in result.errors[0]


def test_warning_page_gives_warning_group(group_folder):
    short = dict(PAGE_OK, transcription="hi")
    result, *_ = _run(group_folder, [json.dumps(short), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])
    assert result.pages[0].status == "warning" and result.status == "warning"


def test_single_page_skips_summary(group_folder):
    (group_folder / "order.json").write_text(json.dumps({"files": ["page1.jpg"]}))
    result, api, _ = _run(group_folder, [json.dumps(PAGE_OK)])
    assert len(api.calls) == 1 and result.summary is None
    assert result.status == "warning"     # page2.jpg on disk but not in order -> group warning


def test_empty_ocr_text_block_is_omitted(group_folder):
    (group_folder / "page1.txt").write_text("")
    _, api, _ = _run(group_folder, [json.dumps(PAGE_OK), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])
    types = [b["type"] for b in api.calls[0][0]["content"]]
    assert types == ["text", "image"]


def test_no_images_is_failed(tmp_path):
    (tmp_path / "n1").mkdir()
    result = nt.process_group("n1", input_dir=str(tmp_path), call_api_fn=lambda m, t: ("", {}), run_id="r1")
    assert result.status == "failed" and "no images" in result.errors[0]


def test_persists_current_and_history_and_metrics(group_folder, tmp_path, monkeypatch):
    current = tmp_path / "cur.json"
    history = tmp_path / "hist.json"
    metrics = tmp_path / "m.jsonl"
    monkeypatch.setattr(nt, "CURRENT_RESPONSES_FILE", str(current))
    monkeypatch.setattr(nt, "RESPONSES_FILE", str(history))
    monkeypatch.setattr("pipeline_utils.METRICS_FILE", str(metrics))
    _run(group_folder, [json.dumps(PAGE_OK), "{a", json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])
    cur = json.loads(current.read_text())
    (key, group), = cur.items()
    assert group["status"] == "done" and group["pages"][1]["attempts"] == 2
    assert key in json.loads(history.read_text())
    rows = [json.loads(l) for l in metrics.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["page", "page", "summary"]
    assert rows[1]["attempts"] == 2 and rows[1]["input_tokens"] == 20


def test_history_key_dedups_on_rerun(group_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(nt, "CURRENT_RESPONSES_FILE", str(tmp_path / "cur.json"))
    monkeypatch.setattr(nt, "RESPONSES_FILE", str(tmp_path / "hist.json"))
    script = [json.dumps(PAGE_OK), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)]
    _run(group_folder, list(script))
    _run(group_folder, list(script))
    assert len(json.loads((tmp_path / "hist.json").read_text())) == 2
