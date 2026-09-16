import json
from googlevision_translater import process_group


def test_process_group_writes_sidecars_in_order(group_folder):
    seen = []

    def fake_extract(path):
        seen.append(path.split("/")[-1])
        return f"text for {seen[-1]}"

    r = process_group("n1", input_dir=str(group_folder.parent), extract_fn=fake_extract)
    assert r.ok
    assert seen == ["page1.jpg", "page2.jpg"]
    assert (group_folder / "page1.txt").read_text() == "text for page1.jpg"


def test_process_group_remaps_converted_png(group_folder):
    (group_folder / "order.json").write_text(json.dumps({"files": ["page1.png", "page2.jpg"]}))
    r = process_group("n1", input_dir=str(group_folder.parent), extract_fn=lambda p: "t")
    assert r.ok and (group_folder / "page1.txt").exists()


def test_process_group_reports_extract_failure(group_folder):
    def boom(path):
        raise RuntimeError("vision down")
    r = process_group("n1", input_dir=str(group_folder.parent), extract_fn=boom)
    assert not r.ok and "vision down" in r.error and "page1.jpg" in r.error


def test_process_group_missing_folder(tmp_path):
    r = process_group("n9", input_dir=str(tmp_path), extract_fn=lambda p: "t")
    assert not r.ok
