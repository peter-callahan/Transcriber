import json
import pytest
from conftest import PAGE_OK, SUMMARY_OK

from pipeline_utils import ParseError, StructureError
import note_translater as nt


# --- moved helpers ---------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("2025_08_01", "2025_08_01"),
    ("2025-08-01", "2025_08_01"),
    ("Aug 1, 2025", "2025_08_01"),
    ("Aug 2025", "2025_08_01"),
    ("garbage", None),
])
def test_parse_date_string(raw, expected):
    assert nt.parse_date_string(raw) == expected


def test_clean_json_text_strips_fences_and_escapes_newlines():
    raw = '```json\n{"transcription": "line one\nline two"}\n```'
    assert json.loads(nt.clean_json_text(raw)) == {"transcription": "line one\nline two"}


def test_generate_uuid_is_stable_and_order_independent():
    assert nt.generate_uuid(["b.jpg", "a.jpg"], "m") == nt.generate_uuid(["a.jpg", "b.jpg"], "m")
    with pytest.raises(ValueError):
        nt.generate_uuid([], "m")


# --- parsers --------------------------------------------------------------

def test_parse_page_ok():
    data = nt.parse_page(json.dumps(PAGE_OK))
    assert data["transcription"].startswith("Aug 1")
    assert data["uncertain"] == []


def test_parse_page_defaults_missing_uncertain_to_empty():
    d = {k: v for k, v in PAGE_OK.items() if k != "uncertain"}
    assert nt.parse_page(json.dumps(d))["uncertain"] == []


def test_parse_page_bad_json_raises_parse_error():
    with pytest.raises(ParseError):
        nt.parse_page("{nope")


@pytest.mark.parametrize("missing", ["title", "date", "transcription", "tags"])
def test_parse_page_missing_field_raises_structure_error(missing):
    d = {k: v for k, v in PAGE_OK.items() if k != missing}
    with pytest.raises(StructureError, match=missing):
        nt.parse_page(json.dumps(d))


def test_parse_page_malformed_uncertain_raises_structure_error():
    d = dict(PAGE_OK, uncertain=[{"context": "no text key"}])
    with pytest.raises(StructureError, match="uncertain"):
        nt.parse_page(json.dumps(d))


def test_parse_page_rejects_non_dict():
    with pytest.raises(StructureError):
        nt.parse_page("[1, 2]")


def test_parse_summary_ok_and_missing():
    assert nt.parse_summary(json.dumps(SUMMARY_OK))["title"] == "Two page note"
    d = {k: v for k, v in SUMMARY_OK.items() if k != "continuous_transcription"}
    with pytest.raises(StructureError, match="continuous_transcription"):
        nt.parse_summary(json.dumps(d))


# --- warnings -------------------------------------------------------------

def test_page_warnings_clean():
    assert nt.page_warnings(PAGE_OK) == []


def test_page_warnings_inline_marker_is_warning_not_error():
    d = dict(PAGE_OK, transcription="Eggs, [?milk] and bread for the whole week ahead")
    w = nt.page_warnings(d)
    assert any("inline uncertainty marker" in x for x in w)


def test_page_warnings_short_and_too_many_tags():
    d = dict(PAGE_OK, transcription="hi", tags=["a", "b", "c", "d"])
    w = nt.page_warnings(d)
    assert any("short" in x for x in w) and any("tags" in x for x in w)


# --- group validation -----------------------------------------------------

def test_validate_group_flags_order_mismatch(group_folder):
    w = nt.validate_group(str(group_folder), ["page1.jpg"], None)
    assert any("not in order list" in x for x in w)


def test_validate_group_flags_missing_order_json(group_folder):
    (group_folder / "order.json").unlink()
    w = nt.validate_group(str(group_folder), ["page1.jpg", "page2.jpg"], None)
    assert any("No order.json" in x for x in w)


def test_validate_group_flags_summary_problems(group_folder):
    s = dict(SUMMARY_OK, tags=["a", "b", "c", "d"], continuous_transcription="x")
    w = nt.validate_group(str(group_folder), ["page1.jpg", "page2.jpg"], s)
    assert len(w) == 2


# --- prompts --------------------------------------------------------------

def test_single_prompt_mentions_uncertain_field_and_hash_changes_with_tags():
    p = nt.build_single_prompt([])
    assert '"uncertain"' in p and "Do not mark" in p
    assert nt.prompt_hash(p) != nt.prompt_hash(nt.build_single_prompt(["x"]))
