import json
import os
from datetime import datetime, timezone, timedelta
import yaml
from conftest import PAGE_OK, SUMMARY_OK, make_jpeg

from export_responses import build_markdown, export_run, sanitize_filename

FIXED_TIME = datetime(2026, 9, 20, 16, 53, 8, tzinfo=timezone(timedelta(hours=-4)))


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


def _frontmatter(md):
    """Parse the YAML frontmatter block out of rendered markdown."""
    assert md.startswith("---\n")
    _, fm_text, _ = md.split("---\n", 2)
    return yaml.safe_load(fm_text)


def test_single_page_markdown(tmp_path):
    folder, md = build_markdown(_group(tmp_path), exported_at=FIXED_TIME)
    assert folder == "2025_08_01 - Grocery list"
    fm = _frontmatter(md)
    assert fm == {
        "title": "Grocery list",
        "date created": "2025_08_01",
        "date modified": "2026_09_20 16:53:08 -04:00",
        "tags": ["errands"],
        "document_type": "journal entry",
        "source": "digital-conversion",
    }
    assert PAGE_OK["transcription"] in md
    assert "uncertain" not in md and "milk\"" not in md
    # no leftover inline metadata lines from the old rendering
    assert "# Grocery list" not in md and "**Date:**" not in md and "**Tags:**" not in md


def test_multi_page_uses_summary(tmp_path):
    folder, md = build_markdown(_group(tmp_path, pages=2), exported_at=FIXED_TIME)
    assert folder == "2025_08_01 - Two page note"
    assert "## Summary" in md and SUMMARY_OK["continuous_transcription"] in md
    fm = _frontmatter(md)
    assert fm["title"] == "Two page note"
    assert fm["tags"] == ["errands", "planning"]


def test_partial_date_is_normalized_mechanically(tmp_path):
    """No LLM call — parse_date_string() coerces 'August 2020' to day 01."""
    group = _group(tmp_path)
    group["pages"][0]["data"] = dict(PAGE_OK, date="August 2020")
    folder, md = build_markdown(group, exported_at=FIXED_TIME)
    assert folder == "2020_08_01 - Grocery list"
    assert _frontmatter(md)["date created"] == "2020_08_01"


def test_unparseable_date_falls_back_to_raw_string(tmp_path):
    group = _group(tmp_path)
    group["pages"][0]["data"] = dict(PAGE_OK, date="sometime last spring")
    _, md = build_markdown(group, exported_at=FIXED_TIME)
    assert _frontmatter(md)["date created"] == "sometime last spring"


def test_empty_tags_render_as_empty_array(tmp_path):
    group = _group(tmp_path)
    group["pages"][0]["data"] = dict(PAGE_OK, tags=[])
    _, md = build_markdown(group, exported_at=FIXED_TIME)
    assert _frontmatter(md)["tags"] == []
    assert "tags: []" in md


def test_tags_are_always_quoted(tmp_path):
    """Consistent quoting regardless of whether a tag would need it (e.g. a purely
    numeric tag like '2024' needs quotes to stay a string; plain words don't)."""
    group = _group(tmp_path)
    group["pages"][0]["data"] = dict(PAGE_OK, tags=["errands", "2024"])
    _, md = build_markdown(group, exported_at=FIXED_TIME)
    assert "tags: ['errands', '2024']" in md
    assert _frontmatter(md)["tags"] == ["errands", "2024"]


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
