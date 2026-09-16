import json
from PIL import Image

import process_images
from process_images import process_group, resize_image


def test_resize_image_shrinks_and_converts_png(tmp_path):
    src = tmp_path / "big.png"
    Image.new("RGB", (3000, 1000)).save(src, "PNG")
    out = resize_image(str(src))
    assert out == str(tmp_path / "big.jpg")
    assert not src.exists()
    with Image.open(out) as img:
        assert img.size[0] == 2000 and img.size[1] in (666, 667) and img.format == "JPEG"


def test_resize_image_raises_on_garbage(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    try:
        resize_image(str(bad))
    except Exception:
        return
    raise AssertionError("expected resize_image to raise")


def test_process_group_ok(group_folder):
    r = process_group("n1", input_dir=str(group_folder.parent))
    assert r.ok and r.error is None


def test_process_group_missing_folder(tmp_path):
    r = process_group("n9", input_dir=str(tmp_path))
    assert not r.ok and "not found" in r.error


def test_process_group_reports_bad_file(group_folder):
    (group_folder / "page2.jpg").write_bytes(b"garbage")
    r = process_group("n1", input_dir=str(group_folder.parent))
    assert not r.ok and "page2.jpg" in r.error
