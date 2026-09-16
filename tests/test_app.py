import json
import os
import pytest
from conftest import make_jpeg

import app as app_module
from pipeline_utils import StageResult, GroupResult, PageResult


@pytest.fixture
def client(tmp_path, monkeypatch):
    temp = tmp_path / "temp"
    inp = tmp_path / "input"
    temp.mkdir()
    monkeypatch.setattr(app_module, "TEMP_FOLDER", str(temp))
    monkeypatch.setattr(app_module, "INPUT_FOLDER", str(inp))
    monkeypatch.setattr(app_module, "OUTPUT_FOLDER", str(tmp_path / "out"))
    monkeypatch.setattr(app_module, "RUN_STATUS_FILE", str(tmp_path / "run_status.json"))
    monkeypatch.setattr(app_module, "CURRENT_RESPONSES_FILE", str(tmp_path / "cur.json"))
    app_module.processing_progress.clear()
    app_module.processing_progress.update(app_module._new_progress())
    make_jpeg(temp / "a.jpg")
    make_jpeg(temp / "b.jpg")
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _ok_stage(group_name, input_dir=None):
    return StageResult(True)


def _transcribe_factory(fail_groups=()):
    def transcribe(group_name, on_update=None, input_dir=None, run_id=None):
        on_update(group_name, stage="transcribing", pages_total=1, pages_done=0, attempts=0)
        status = "failed" if group_name in fail_groups else "done"
        on_update(group_name, stage="complete", status=status, pages_done=1, attempts=2,
                  error="page a.jpg: boom after 3 attempts" if status == "failed" else None)
        return GroupResult(group_name=group_name, status=status, file_order=["a.jpg"],
                           image_paths=[], pages=[PageResult(filename="a.jpg", status=status)],
                           errors=["page a.jpg: boom after 3 attempts"] if status == "failed" else [])
    return transcribe


def _export(results, output_dir):
    return [{"group_name": r["group_name"], "ok": True, "path": "/x", "error": None} for r in results]


def _post(client, groups):
    return client.post("/api/process", data=json.dumps({"groups": groups}),
                       content_type="application/json")


def test_all_groups_succeed(client, monkeypatch):
    exported = []
    monkeypatch.setitem(app_module.PIPELINE, "resize", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "ocr", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", _transcribe_factory())
    monkeypatch.setitem(app_module.PIPELINE, "export", lambda r, o: exported.extend(r) or _export(r, o))
    resp = _post(client, [{"images": ["a.jpg"]}, {"images": ["b.jpg"]}])
    assert resp.status_code == 200
    progress = client.get("/api/progress").get_json()
    assert progress["status"] == "completed" and progress["percentage"] == 100
    assert [g["status"] for g in progress["groups"]] == ["done", "done"]
    assert progress["groups"][0]["label"] == "1 page"
    assert [r["group_name"] for r in exported] == ["n1", "n2"]
    assert os.path.exists(app_module.RUN_STATUS_FILE)


def test_failed_group_does_not_stop_run(client, monkeypatch):
    exported = []
    monkeypatch.setitem(app_module.PIPELINE, "resize", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "ocr", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", _transcribe_factory(fail_groups=("n1",)))
    monkeypatch.setitem(app_module.PIPELINE, "export", lambda r, o: exported.extend(r) or _export(r, o))
    resp = _post(client, [{"images": ["a.jpg"]}, {"images": ["b.jpg"]}])
    assert resp.status_code == 207
    body = resp.get_json()
    assert body["groups_processed"] == 1 and body["failed_groups"] == ["n1"]
    groups = client.get("/api/progress").get_json()["groups"]
    assert groups[0]["status"] == "failed" and "boom" in groups[0]["error"]
    assert groups[1]["status"] == "done"
    assert [r["group_name"] for r in exported] == ["n2"]


def test_stage_failure_marks_group_failed(client, monkeypatch):
    monkeypatch.setitem(app_module.PIPELINE, "resize", lambda g, input_dir=None: StageResult(False, "corrupt"))
    monkeypatch.setitem(app_module.PIPELINE, "ocr", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", _transcribe_factory())
    monkeypatch.setitem(app_module.PIPELINE, "export", _export)
    resp = _post(client, [{"images": ["a.jpg"]}])
    assert resp.status_code == 500
    g = client.get("/api/progress").get_json()["groups"][0]
    assert g["status"] == "failed" and g["stage"] == "resizing" and "corrupt" in g["error"]


def test_unexpected_exception_is_isolated(client, monkeypatch):
    def explode(group_name, input_dir=None):
        if group_name == "n1":
            raise RuntimeError("kaboom")
        return StageResult(True)
    monkeypatch.setitem(app_module.PIPELINE, "resize", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "ocr", explode)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", _transcribe_factory())
    monkeypatch.setitem(app_module.PIPELINE, "export", _export)
    resp = _post(client, [{"images": ["a.jpg"]}, {"images": ["b.jpg"]}])
    assert resp.status_code == 207
    groups = client.get("/api/progress").get_json()["groups"]
    assert "kaboom" in groups[0]["error"] and groups[1]["status"] == "done"


def test_progress_falls_back_to_run_status_file(client):
    saved = dict(app_module._new_progress(), status="completed", groups=[{"name": "n1", "status": "failed"}])
    with open(app_module.RUN_STATUS_FILE, "w") as f:
        json.dump(saved, f)
    assert client.get("/api/progress").get_json()["groups"][0]["status"] == "failed"


def test_export_failure_is_caught_and_reported(client, monkeypatch):
    """Fix: an exception during export/cleanup (e.g. an unwritable OUTPUT_FOLDER)
    must not propagate out of the route as a bare 500 — it should be caught,
    reflected in processing_progress, and the already-completed per-group
    results must not be lost."""
    def exploding_export(results, output_dir):
        raise OSError("Read-only file system")
    monkeypatch.setitem(app_module.PIPELINE, "resize", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "ocr", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", _transcribe_factory())
    monkeypatch.setitem(app_module.PIPELINE, "export", exploding_export)
    resp = _post(client, [{"images": ["a.jpg"]}])
    assert resp.status_code == 500
    assert "Export/cleanup failed" in resp.get_json()["error"]
    progress = client.get("/api/progress").get_json()
    assert progress["status"] == "error"
    assert "Error during export/cleanup" in progress["current_step"]
    # the group that completed before the export step ran is still reflected
    assert progress["groups"][0]["status"] == "done"


def test_run_start_clears_previous_run_files(client, monkeypatch):
    with open(app_module.CURRENT_RESPONSES_FILE, "w") as f:
        f.write("{}")
    monkeypatch.setitem(app_module.PIPELINE, "resize", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "ocr", _ok_stage)
    seen = {}
    def transcribe(group_name, on_update=None, input_dir=None, run_id=None):
        seen["current_exists_at_start"] = os.path.exists(app_module.CURRENT_RESPONSES_FILE)
        return _transcribe_factory()(group_name, on_update=on_update)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", transcribe)
    monkeypatch.setitem(app_module.PIPELINE, "export", _export)
    _post(client, [{"images": ["a.jpg"]}])
    assert seen["current_exists_at_start"] is False
