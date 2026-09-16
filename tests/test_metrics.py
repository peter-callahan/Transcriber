import json
import pipeline_utils as pu
from pipeline_utils import CallMeta, append_metric, metric_row, atomic_write_json


def _meta(**over):
    base = dict(run_id="r1", group_name="n1", filename="p.jpg", kind="page",
                provider="openai", model="m", prompt_hash="h", attempts=1,
                latency_ms=5, input_tokens=1, output_tokens=1, word_count=2,
                uncertain_count=0, timestamp="2026-09-15T00:00:00")
    base.update(over)
    return CallMeta(**base)


def test_append_metric_writes_one_line_per_call(tmp_path):
    path = tmp_path / "m.jsonl"
    append_metric(metric_row(_meta(), "done", None), path=str(path))
    append_metric(metric_row(_meta(filename="q.jpg"), "failed", "boom"), path=str(path))
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(l) for l in lines)
    assert first["filename"] == "p.jpg" and first["status"] == "done" and first["error"] is None
    assert second["status"] == "failed" and second["error"] == "boom"


def test_append_metric_never_truncates(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"existing": true}\n')
    append_metric(metric_row(_meta(), "done", None), path=str(path))
    assert path.read_text().startswith('{"existing": true}')


def test_atomic_write_json_replaces_file(tmp_path):
    path = tmp_path / "s.json"
    atomic_write_json(str(path), {"a": 1})
    atomic_write_json(str(path), {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    assert list(tmp_path.iterdir()) == [path]          # no temp file left behind
