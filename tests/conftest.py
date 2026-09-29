import os
import sys
import json
import pathlib
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SANDBOX = pathlib.Path(tempfile.mkdtemp(prefix="transcriber-tests-"))
os.environ.update({
    "INPUT_FOLDER": str(SANDBOX / "input_images"),
    "TEMP_FOLDER": str(SANDBOX / "temp_uploads"),
    "OUTPUT_FOLDER": str(SANDBOX / "markdown_output"),
    "RESPONSES_FILE": str(SANDBOX / "responses.json"),
    "CURRENT_RESPONSES_FILE": str(SANDBOX / "responses_current.json"),
    "METRICS_FILE": str(SANDBOX / "metrics.jsonl"),
    "RUN_STATUS_FILE": str(SANDBOX / "run_status.json"),
    "OBSIDIAN_TAGS_FILE": str(SANDBOX / "obsidian_tags.json"),
    "LOG_FILE": str(SANDBOX / "transcriber.log"),
    "AI_PROVIDER": "openai",
    "OPENAI_MODEL": "test-model",
    "GOOGLE_APPLICATION_CREDENTIALS": "",
})

import pytest
from PIL import Image


PAGE_OK = {
    "title": "Grocery list",
    "date": "2025_08_01",
    "transcription": "Aug 1 2025\nEggs, milk, bread and a long list of other things.",
    "tags": ["errands"],
    "uncertain": [],
}

SUMMARY_OK = {
    "title": "Two page note",
    "date": "2025_08_01",
    "summary": "Notes about groceries and a plan.",
    "continuous_transcription": "Aug 1 2025\nEggs, milk, bread and a long list of other things. Then a plan for the week.",
    "tags": ["errands", "planning"],
}


def make_jpeg(path, size=(300, 200)):
    Image.new("RGB", size, color=(200, 200, 200)).save(path, "JPEG")
    return path


@pytest.fixture
def group_folder(tmp_path):
    """A group folder with two JPEG pages, OCR sidecars and order.json."""
    folder = tmp_path / "input_images" / "n1"
    folder.mkdir(parents=True)
    make_jpeg(folder / "page1.jpg")
    make_jpeg(folder / "page2.jpg")
    (folder / "page1.txt").write_text("ocr text one")
    (folder / "page2.txt").write_text("ocr text two")
    (folder / "order.json").write_text(json.dumps({"files": ["page1.jpg", "page2.jpg"]}))
    return folder


class ScriptedApi:
    """Fake call_api: returns scripted results in order. An item that is an
    Exception instance is raised; a str is returned as the model text."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, messages, max_tokens):
        self.calls.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, {"input_tokens": 10, "output_tokens": 5}


@pytest.fixture
def scripted_api():
    return ScriptedApi
