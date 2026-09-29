import json
import pytest
from conftest import make_jpeg

from pipeline_utils import get_file_order, resolve_image_path, parse_date_string


def test_get_file_order_reads_order_json(tmp_path):
    (tmp_path / "order.json").write_text(json.dumps({"files": ["b.jpg", "a.jpg"]}))
    assert get_file_order(str(tmp_path)) == ["b.jpg", "a.jpg"]


def test_get_file_order_falls_back_to_sorted_images(tmp_path):
    for name in ["z.jpg", "a.png", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    assert get_file_order(str(tmp_path)) == ["a.png", "z.jpg"]


def test_get_file_order_bad_json_falls_back(tmp_path):
    (tmp_path / "order.json").write_text("{not json")
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert get_file_order(str(tmp_path)) == ["a.jpg"]


def test_resolve_image_path_exact(tmp_path):
    make_jpeg(tmp_path / "p.jpg")
    assert resolve_image_path(str(tmp_path), "p.jpg") == ("p.jpg", str(tmp_path / "p.jpg"))


def test_resolve_image_path_remaps_png_to_jpg(tmp_path):
    make_jpeg(tmp_path / "p.jpg")
    assert resolve_image_path(str(tmp_path), "p.png") == ("p.jpg", str(tmp_path / "p.jpg"))


def test_resolve_image_path_missing_returns_none(tmp_path):
    assert resolve_image_path(str(tmp_path), "nope.heic") is None


@pytest.mark.parametrize("raw,expected", [
    ("2025_08_01", "2025_08_01"),
    ("2025-08-01", "2025_08_01"),
    ("Aug 1, 2025", "2025_08_01"),
    ("Aug 2025", "2025_08_01"),          # partial date: day coerced to 01
    ("August 2020", "2020_08_01"),       # partial date: day coerced to 01
    ("2020-08", "2020_08_01"),           # partial date: day coerced to 01
    ("2020_08", "2020_08_01"),           # partial date, underscore (the model's own convention)
    ("garbage", None),                   # unparseable: caller falls back to raw string
])
def test_parse_date_string(raw, expected):
    assert parse_date_string(raw) == expected
