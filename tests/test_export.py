import json
import os
from conftest import PAGE_OK, SUMMARY_OK, make_jpeg

from export_responses import build_markdown, export_run, sanitize_filename


def _group(tmp_path, status="done", pages=1, summary=True):
    img_paths = []
    for i in range(pages):
        p = tmp_path / f"img{i}.jpg"
        make_jpeg(p)
        img_paths.append(str(p))
    return {
        "group_name": "n1", "status": status,
        "file_order": [os.path.basename(p) for p in img_paths], "image_paths": img_paths,
        "pages": [{"filename": os.path.basename(p), "status": "done", "data": dict(PAGE_OK),
                   "uncertain": [{"text": "milk", "context": "Eggs, milk"}]} for p in img_paths],
        "summary": dict(SUMMARY_OK) if (summary and pages > 1) else None,
        "warnings": [], "errors": [],
    }


def test_sanitize_filename():
    assert sanitize_filename('a/b:c?') == "a_b_c"
    assert sanitize_filename("") == "Untitled"


def test_single_page_markdown(tmp_path):
    folder, md = build_markdown(_group(tmp_path))
    assert folder == "2025_08_01 - Grocery list"
    assert md.startswith("# Grocery list\n\n**Date:** 2025_08_01\n\n**Tags:** #errands\n\n")
    assert PAGE_OK["transcription"] in md
    assert "uncertain" not in md and "milk\"" not in md


def test_multi_page_uses_summary(tmp_path):
    folder, md = build_markdown(_group(tmp_path, pages=2))
    assert folder == "2025_08_01 - Two page note"
    assert "## Summary" in md and SUMMARY_OK["continuous_transcription"] in md


def test_export_run_writes_folders_and_images(tmp_path):
    out = tmp_path / "out"
    r = export_run([_group(tmp_path, pages=2)], str(out))
    assert r == [{"group_name": "n1", "ok": True, "path": str(out / "2025_08_01 - Two page note"), "error": None}]
    folder = out / "2025_08_01 - Two page note"
    assert (folder / "2025_08_01 - Two page note.md").exists()
    assert sorted(os.listdir(folder / "images")) == ["img0.jpg", "img1.jpg"]


def test_export_run_skips_failed_and_dedups_names(tmp_path):
    out = tmp_path / "out"
    export_run([_group(tmp_path)], str(out))
    r = export_run([_group(tmp_path), dict(_group(tmp_path), status="failed", group_name="n2")], str(out))
    assert [x["group_name"] for x in r] == ["n1"]
    assert r[0]["path"].endswith("2025_08_01 - Grocery list_2")


def test_export_run_isolates_a_bad_group(tmp_path, monkeypatch):
    out = tmp_path / "out"
    bad = dict(_group(tmp_path), group_name="n2", pages=[])
    bad["summary"] = None
    r = export_run([_group(tmp_path), bad], str(out))
    assert r[0]["ok"] and not r[1]["ok"] and r[1]["error"]
