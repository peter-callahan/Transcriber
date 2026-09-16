from pipeline_utils import (
    StageResult, CallMeta, PageResult, GroupResult, rollup_status,
)


def test_rollup_failed_when_any_page_failed():
    assert rollup_status(["done", "failed"], None, []) == "failed"


def test_rollup_failed_when_summary_failed():
    assert rollup_status(["done", "done"], "boom", []) == "failed"


def test_rollup_failed_when_no_pages():
    assert rollup_status([], None, []) == "failed"


def test_rollup_warning_when_any_page_warning():
    assert rollup_status(["done", "warning"], None, []) == "warning"


def test_rollup_warning_when_group_warning():
    assert rollup_status(["done"], None, ["No order.json found"]) == "warning"


def test_rollup_done():
    assert rollup_status(["done", "done"], None, []) == "done"


def test_group_result_to_dict_is_json_shaped():
    meta = CallMeta(
        run_id="r1", group_name="n1", filename="p.jpg", kind="page",
        provider="openai", model="m", prompt_hash="abc", attempts=1,
        latency_ms=12, input_tokens=10, output_tokens=5, word_count=3,
        uncertain_count=0, timestamp="2026-09-15T00:00:00",
    )
    page = PageResult(filename="p.jpg", status="done", attempts=1,
                      data={"transcription": "a b c"}, meta=meta)
    group = GroupResult(group_name="n1", status="done", file_order=["p.jpg"],
                        image_paths=["/x/p.jpg"], pages=[page])
    d = group.to_dict()
    assert d["pages"][0]["meta"]["prompt_hash"] == "abc"
    assert d["summary"] is None
    assert d["pages"][0]["uncertain"] == []


def test_stage_result_defaults():
    assert StageResult(True).error is None
