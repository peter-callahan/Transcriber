"""End-to-end integration test across the note_translater -> export_responses boundary.

Every other test in this suite monkeypatches PIPELINE or call_api_fn/extract_fn to
avoid running real pipeline stages together. This test is the exception: it exercises
the REAL note_translater.process_group() output flowing straight into the REAL
export_responses.export_run(), with only the outermost AI API call faked (via the
ScriptedApi fixture already used throughout tests/test_note_translater.py). The point
is to catch a field-name typo or shape mismatch at this boundary that every mocked
test would miss.
"""
import json
import os

from conftest import PAGE_OK, SUMMARY_OK, ScriptedApi

import note_translater as nt
import export_responses


def test_real_process_group_output_exports_cleanly(group_folder, tmp_path):
    # Only the AI API call is faked; everything else (parsing, retry, GroupResult
    # construction, markdown building, file writing) is real production code.
    api = ScriptedApi([json.dumps(PAGE_OK), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])

    result = nt.process_group(
        "n1", input_dir=str(group_folder.parent),
        call_api_fn=api, run_id="r1", sleep_fn=lambda s: None,
    )
    assert result.status == "done"

    # No reshaping: the real GroupResult.to_dict() output goes straight into export_run.
    result_dict = result.to_dict()

    output_dir = tmp_path / "export_out"
    report = export_responses.export_run([result_dict], str(output_dir))

    assert len(report) == 1
    assert report[0]["ok"] is True
    assert report[0]["error"] is None

    expected_folder = output_dir / "2025_08_01 - Two page note"
    assert report[0]["path"] == str(expected_folder)
    assert expected_folder.is_dir()

    md_path = expected_folder / "2025_08_01 - Two page note.md"
    assert md_path.exists()
    content = md_path.read_text()
    assert content.startswith("---\ntitle: Two page note\n")
    assert "date created: '2025_08_01'" in content
    assert SUMMARY_OK["continuous_transcription"] in content

    images_dir = expected_folder / "images"
    assert images_dir.is_dir()
    assert sorted(os.listdir(images_dir)) == ["page1.jpg", "page2.jpg"]
