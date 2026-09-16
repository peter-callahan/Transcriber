# Resilient Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transient API failures are retried within a 3-call cap, a failed group never stops the run, and the user sees per-group status live and after the run, with every API call logged to a permanent metrics file.

**Architecture:** The three pipeline scripts become importable modules with a `process_group()` entry point each, sharing `pipeline_utils.py` (file ordering, provider client, retry loop, result dataclasses, metrics). `app.py` calls them in-process and pushes per-group status into the existing `/api/progress` dict, which is mirrored to `run_status.json`. Export only touches `done`/`warning` groups.

**Tech Stack:** Python 3.11, Flask 2.3, pytest, OpenAI / Anthropic / boto3 (Bedrock) SDKs, Google Cloud Vision, Pillow.

**Spec:** `docs/superpowers/specs/2026-09-15-resilient-pipeline-design.md`

## Global Constraints

- Python interpreter and pip for this project (pyenv virtualenv `media_handler`, 3.11.8):
  - `PY=~/.pyenv/versions/3.11.8/envs/media_handler/bin/python`
  - `PIP=~/.pyenv/versions/3.11.8/envs/media_handler/bin/pip`
  - Run tests with `$PY -m pytest` from the project root. Never use the global `python`/`pip`.
- Retry cap is **3 total API calls** per page and per summary call. Backoff for transport errors: 1s, 4s, 10s (capped).
- Retry triggers: transport errors, unparseable JSON, missing/invalid required fields. Quality warnings (short text, too many tags, inline marker) are **never** retried.
- No inline uncertainty markers in transcription text. `uncertain` is a separate JSON array. Export never touches it.
- Retention: last run only. `run_status.json` and `responses_current.json` persist until the next `POST /api/process`. `metrics.jsonl` is never truncated.
- No result cache. No run history folders. No `tenacity`.
- Exported markdown contains only transcription text plus the existing title/date/tags/summary header.
- Commit after every task with a short imperative message. End every commit message with:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`
- The working tree is clean at start (HEAD `e380815`). Old hyphenated files are removed only in Task 13, after everything imports from the new modules.

## File Map

| File | Status | Responsibility |
|---|---|---|
| `pipeline_utils.py` | create | Shared: constants, logger, `get_file_order`, `resolve_image_path`, result dataclasses, provider client + `call_api`, `is_transient`, `call_with_retry`, `retry_with_feedback`, `append_metric`, `atomic_write_json` |
| `process_images.py` | rewrite | `resize_image`, `process_group(group_name, input_dir=None) -> StageResult`, CLI |
| `googlevision_translater.py` | create (from `googlevision-translater.py`) | `extract_text_from_image(path) -> str`, `process_group(group_name, input_dir=None, extract_fn=None) -> StageResult`, CLI |
| `note_translater.py` | create (from `gpt4-note-translater.py`) | prompts, `parse_page`, `parse_summary`, `page_warnings`, `validate_group`, `process_group(...) -> GroupResult`, persistence, CLI |
| `export_responses.py` | rewrite | `build_markdown(group) -> (folder_name, content)`, `export_run(results, output_dir) -> list[dict]`, CLI |
| `app.py` | modify | In-process orchestration, `groups` status array, `run_status.json`, `/api/progress` fallback |
| `templates/index.html` | modify | Group status list, confirm-before-run, render last run on load |
| `tests/conftest.py`, `tests/test_*.py` | create | pytest suite, no live API calls |
| `.gitignore`, `requirements.txt`, `transpose_notes.sh`, `README.md`, `TODOS.md` | modify | housekeeping |
| `gpt4-note-translater.py`, `googlevision-translater.py` | delete (Task 13) | replaced |

---

### Task 1: Test scaffolding and `pipeline_utils` file helpers

**Files:**
- Create: `pipeline_utils.py`
- Create: `tests/conftest.py`
- Create: `tests/test_utils.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `pipeline_utils.IMAGE_EXTENSIONS: tuple[str, ...]`, `pipeline_utils.INPUT_FOLDER: str`, `pipeline_utils.logger`, `get_file_order(folder_path: str) -> list[str]`, `resolve_image_path(folder_path: str, image_file: str) -> tuple[str, str] | None` (returns `(filename, absolute_path)` with PNG/HEIC→JPG remap, or `None` when neither exists).

- [ ] **Step 1: Install pytest into the project virtualenv and record it**

Run: `~/.pyenv/versions/3.11.8/envs/media_handler/bin/pip install pytest`

Append to `requirements.txt` (the file currently has no trailing newline — add one first):

```
pytest>=8.0
```

- [ ] **Step 2: Write `tests/conftest.py`**

This sets every path the modules read from the environment to a throwaway sandbox *before* any module is imported, so `load_dotenv()` (which does not override existing variables) cannot point tests at real folders.

```python
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
```

- [ ] **Step 3: Write the failing tests for `get_file_order` and `resolve_image_path`**

`tests/test_utils.py`:

```python
import json
import pytest
from conftest import make_jpeg

from pipeline_utils import get_file_order, resolve_image_path


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
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline_utils'`

- [ ] **Step 5: Create `pipeline_utils.py` with the file helpers**

```python
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transcriber")

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.heic')
INPUT_FOLDER = os.path.expanduser(os.getenv('INPUT_FOLDER', 'input_images'))


def get_file_order(folder_path):
    """Files in user-defined order from order.json; sorted image files as fallback."""
    order_file = os.path.join(folder_path, 'order.json')
    if os.path.exists(order_file):
        try:
            with open(order_file, 'r') as f:
                return json.load(f).get('files', [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to read order.json: {e}, falling back to sorted order")
    return sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def resolve_image_path(folder_path, image_file):
    """(filename, path) for image_file, remapping to the .jpg that process_images
    produces for PNG/HEIC inputs. None if neither exists."""
    path = os.path.join(folder_path, image_file)
    if os.path.isfile(path):
        return image_file, path
    jpg_name = os.path.splitext(image_file)[0] + '.jpg'
    jpg_path = os.path.join(folder_path, jpg_name)
    if os.path.isfile(jpg_path):
        return jpg_name, jpg_path
    return None
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_utils.py -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add pipeline_utils.py tests/conftest.py tests/test_utils.py requirements.txt
git commit -m "add pipeline_utils with shared file helpers and pytest scaffolding

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Result dataclasses and status roll-up

**Files:**
- Modify: `pipeline_utils.py`
- Create: `tests/test_results.py`

**Interfaces:**
- Produces (all in `pipeline_utils`):
  - `StageResult(ok: bool, error: str | None = None)`
  - `CallMeta(run_id, group_name, filename, kind, provider, model, prompt_hash, attempts, latency_ms, input_tokens, output_tokens, word_count, uncertain_count, timestamp)` — all fields as in the spec; `filename` is `None` for the summary call; `kind` is `"page"` or `"summary"`.
  - `PageResult(filename, status, attempts=0, data=None, uncertain=[], warnings=[], error=None, history=[], meta=None)`
  - `GroupResult(group_name, status, file_order, image_paths, pages, summary=None, summary_attempts=0, summary_meta=None, summary_error=None, warnings=[], errors=[])` with `.to_dict() -> dict` (dataclasses converted recursively).
  - `rollup_status(page_statuses: list[str], summary_error: str | None, group_warnings: list[str]) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_results.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_results.py -v`
Expected: FAIL with `ImportError: cannot import name 'StageResult'`

- [ ] **Step 3: Add the dataclasses to `pipeline_utils.py`**

Append after `resolve_image_path`:

```python
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class StageResult:
    ok: bool
    error: Optional[str] = None


@dataclass
class CallMeta:
    run_id: str
    group_name: str
    filename: Optional[str]
    kind: str                       # "page" | "summary"
    provider: str
    model: str
    prompt_hash: str
    attempts: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    word_count: int
    uncertain_count: int
    timestamp: str


@dataclass
class PageResult:
    filename: str
    status: str                     # "done" | "warning" | "failed"
    attempts: int = 0
    data: Optional[dict] = None
    uncertain: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    error: Optional[str] = None
    history: list = field(default_factory=list)
    meta: Optional[CallMeta] = None


@dataclass
class GroupResult:
    group_name: str
    status: str                     # "done" | "warning" | "failed"
    file_order: list
    image_paths: list
    pages: list                     # list[PageResult]
    summary: Optional[dict] = None
    summary_attempts: int = 0
    summary_meta: Optional[CallMeta] = None
    summary_error: Optional[str] = None
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def rollup_status(page_statuses, summary_error, group_warnings):
    if not page_statuses or "failed" in page_statuses or summary_error:
        return "failed"
    if "warning" in page_statuses or group_warnings:
        return "warning"
    return "done"
```

Move the two `import` lines to the top of the file with the other imports.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_results.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline_utils.py tests/test_results.py
git commit -m "add pipeline result dataclasses and status roll-up

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Provider layer — message translation, `call_api`, transient detection

**Files:**
- Modify: `pipeline_utils.py`
- Create: `tests/test_provider.py`

**Interfaces:**
- Produces (in `pipeline_utils`):
  - Internal message shape: `[{"role": "user"|"assistant", "content": [ {"type": "text", "text": str} | {"type": "image", "base64": str} ]}]`
  - `get_provider() -> str` (`openai` default, `anthropic`, `bedrock`), `get_model() -> str`, `get_client()` (lazy, cached)
  - `to_provider_messages(messages, provider) -> list[dict]`
  - `call_api(messages, max_tokens) -> tuple[str, dict]` where dict is `{"input_tokens": int, "output_tokens": int}`
  - `is_transient(exc: BaseException) -> bool`

- [ ] **Step 1: Write the failing tests**

`tests/test_provider.py`:

```python
import base64
import types
import pytest

import pipeline_utils as pu

IMG = base64.b64encode(b"jpegbytes").decode()
MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "base64": IMG}]},
    {"role": "assistant", "content": [{"type": "text", "text": "{bad"}]},
    {"role": "user", "content": [{"type": "text", "text": "fix it"}]},
]


def test_openai_translation():
    out = pu.to_provider_messages(MESSAGES, "openai")
    assert out[0]["content"][1] == {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{IMG}"}}
    assert out[1] == {"role": "assistant", "content": [{"type": "text", "text": "{bad"}]}


def test_anthropic_translation():
    out = pu.to_provider_messages(MESSAGES, "anthropic")
    assert out[0]["content"][1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": IMG}
    assert out[2]["role"] == "user"


def test_bedrock_translation_decodes_bytes():
    out = pu.to_provider_messages(MESSAGES, "bedrock")
    assert out[0]["content"][0] == {"text": "hi"}
    assert out[0]["content"][1]["image"]["source"]["bytes"] == b"jpegbytes"
    assert out[1]["content"] == [{"text": "{bad"}]


class _Usage:
    prompt_tokens = 7
    completion_tokens = 3
    input_tokens = 7
    output_tokens = 3


def test_call_api_openai_returns_text_and_usage(monkeypatch):
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, max_tokens):
                    msg = types.SimpleNamespace(content="hello")
                    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=_Usage())
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setattr(pu, "get_client", lambda: Client())
    text, usage = pu.call_api(MESSAGES, 100)
    assert text == "hello"
    assert usage == {"input_tokens": 7, "output_tokens": 3}


def test_call_api_anthropic_returns_text_and_usage(monkeypatch):
    class Client:
        class messages:
            @staticmethod
            def create(model, messages, max_tokens):
                return types.SimpleNamespace(content=[types.SimpleNamespace(text="hey")], usage=_Usage())
    monkeypatch.setenv("AI_PROVIDER", "anthropic")
    monkeypatch.setattr(pu, "get_client", lambda: Client())
    assert pu.call_api(MESSAGES, 100) == ("hey", {"input_tokens": 7, "output_tokens": 3})


def test_call_api_bedrock_returns_text_and_usage(monkeypatch):
    class Client:
        @staticmethod
        def converse(modelId, messages, inferenceConfig):
            return {"output": {"message": {"content": [{"text": "yo"}]}},
                    "usage": {"inputTokens": 7, "outputTokens": 3}}
    monkeypatch.setenv("AI_PROVIDER", "bedrock")
    monkeypatch.setattr(pu, "get_client", lambda: Client())
    assert pu.call_api(MESSAGES, 100) == ("yo", {"input_tokens": 7, "output_tokens": 3})


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    status_code = 401


class StatusError(Exception):
    def __init__(self, code):
        self.status_code = code


class ClientError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


@pytest.mark.parametrize("exc,expected", [
    (RateLimitError("slow down"), True),
    (StatusError(503), True),
    (StatusError(429), True),
    (ClientError("ThrottlingException"), True),
    (AuthenticationError("bad key"), False),
    (StatusError(400), False),
    (ClientError("ValidationException"), False),
    (ValueError("nope"), False),
])
def test_is_transient(exc, expected):
    assert pu.is_transient(exc) is expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_provider.py -v`
Expected: FAIL with `AttributeError: module 'pipeline_utils' has no attribute 'to_provider_messages'`

- [ ] **Step 3: Add the provider layer to `pipeline_utils.py`**

Add `import base64` at the top. Append:

```python
# --- Provider layer ---------------------------------------------------------

_client = None
_client_provider = None


def get_provider():
    return os.getenv('AI_PROVIDER', 'openai').lower()


def get_model():
    provider = get_provider()
    if provider == 'anthropic':
        return os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-6')
    if provider == 'bedrock':
        return os.getenv('BEDROCK_MODEL', '')
    return os.getenv('OPENAI_MODEL', 'gpt-4o')


def get_client():
    """SDK client for the configured provider, created on first use."""
    global _client, _client_provider
    provider = get_provider()
    if _client is not None and _client_provider == provider:
        return _client
    if provider == 'anthropic':
        import anthropic
        _client = anthropic.Anthropic()
    elif provider == 'bedrock':
        import boto3
        from botocore.config import Config as BotocoreConfig
        _client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            config=BotocoreConfig(read_timeout=300, connect_timeout=10),
        )
    else:
        import openai
        _client = openai.OpenAI()
    _client_provider = provider
    logger.info(f"Using AI provider: {provider}, model: {get_model()}")
    return _client


def to_provider_messages(messages, provider):
    out = []
    for message in messages:
        blocks = []
        for block in message["content"]:
            if block["type"] == "text":
                if provider == 'bedrock':
                    blocks.append({"text": block["text"]})
                else:
                    blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                if provider == 'anthropic':
                    blocks.append({"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": block["base64"]}})
                elif provider == 'bedrock':
                    blocks.append({"image": {"format": "jpeg", "source": {
                        "bytes": base64.b64decode(block["base64"])}}})
                else:
                    blocks.append({"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{block['base64']}"}})
        out.append({"role": message["role"], "content": blocks})
    return out


def call_api(messages, max_tokens):
    """Send a provider-neutral message list. Returns (text, usage)."""
    provider = get_provider()
    client = get_client()
    model = get_model()
    provider_messages = to_provider_messages(messages, provider)

    if provider == 'anthropic':
        r = client.messages.create(model=model, messages=provider_messages, max_tokens=max_tokens)
        return r.content[0].text, {"input_tokens": r.usage.input_tokens,
                                   "output_tokens": r.usage.output_tokens}
    if provider == 'bedrock':
        r = client.converse(modelId=model, messages=provider_messages,
                            inferenceConfig={"maxTokens": max_tokens})
        usage = r.get('usage', {})
        return r['output']['message']['content'][0]['text'], {
            "input_tokens": usage.get('inputTokens', 0),
            "output_tokens": usage.get('outputTokens', 0)}
    r = client.chat.completions.create(model=model, messages=provider_messages, max_tokens=max_tokens)
    return r.choices[0].message.content, {
        "input_tokens": getattr(r.usage, 'prompt_tokens', 0),
        "output_tokens": getattr(r.usage, 'completion_tokens', 0)}


TRANSIENT_EXCEPTION_NAMES = {
    "RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
    "ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError", "ConnectionClosedError",
}
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
TRANSIENT_AWS_CODES = {
    "ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException",
    "InternalServerException", "ModelNotReadyException",
}


def is_transient(exc):
    """True for rate limits, timeouts, connection drops and 5xx from any provider SDK."""
    if type(exc).__name__ in TRANSIENT_EXCEPTION_NAMES:
        return True
    if getattr(exc, "status_code", None) in TRANSIENT_STATUS_CODES:
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict) and response.get("Error", {}).get("Code") in TRANSIENT_AWS_CODES:
        return True
    return False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_provider.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline_utils.py tests/test_provider.py
git commit -m "add provider-neutral messages, call_api with usage, transient error detection

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Retry loop with repair turns

**Files:**
- Modify: `pipeline_utils.py`
- Create: `tests/test_retry.py`

**Interfaces:**
- Consumes: `call_api`, `is_transient` from Task 3.
- Produces (in `pipeline_utils`):
  - `class ParseError(Exception)`, `class StructureError(Exception)`
  - `Attempt(ok: bool, parsed: dict | None, attempts: int, history: list[dict], error: str | None, usage: dict, latency_ms: int, raw_text: str | None)`
  - `REPAIR_TEMPLATE: str`, `BACKOFF_SECONDS = (1, 4, 10)`
  - `call_with_retry(messages, max_tokens, parse, max_calls=3, call_api_fn=None, sleep_fn=time.sleep) -> Attempt`
  - `retry_with_feedback(messages, prior_output, feedback, max_tokens, parse, **kwargs) -> Attempt`

- [ ] **Step 1: Write the failing tests**

`tests/test_retry.py`:

```python
import json
import pytest

import pipeline_utils as pu
from pipeline_utils import call_with_retry, retry_with_feedback, ParseError, StructureError


class RateLimitError(Exception):
    pass


def parse(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(str(e))
    if "transcription" not in data:
        raise StructureError("missing field: transcription")
    return data


MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
GOOD = json.dumps({"transcription": "ok"})


def test_success_first_call(scripted_api):
    api = scripted_api([GOOD])
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert a.ok and a.attempts == 1 and a.parsed == {"transcription": "ok"}
    assert a.usage == {"input_tokens": 10, "output_tokens": 5}
    assert a.history == []


def test_transport_error_then_success_backs_off(scripted_api):
    api = scripted_api([RateLimitError("429"), GOOD])
    sleeps = []
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=sleeps.append)
    assert a.ok and a.attempts == 2
    assert sleeps == [1]
    assert a.history == [{"kind": "transport", "error": "429"}]
    assert api.calls[1] == MESSAGES          # identical redo, no repair turn


def test_bad_json_gets_repair_turn(scripted_api):
    api = scripted_api(["{oops", GOOD])
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert a.ok and a.attempts == 2
    second = api.calls[1]
    assert second[0] == MESSAGES[0]
    assert second[1] == {"role": "assistant", "content": [{"type": "text", "text": "{oops"}]}
    assert second[2]["role"] == "user"
    repair = second[2]["content"][0]["text"]
    assert "could not be used" in repair and "Do not change" in repair
    assert a.history[0]["kind"] == "parse"


def test_missing_field_is_structure_kind(scripted_api):
    api = scripted_api([json.dumps({"title": "x"}), GOOD])
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert a.ok and a.history[0] == {"kind": "structure", "error": "missing field: transcription"}


def test_three_failures_gives_up(scripted_api):
    api = scripted_api(["{a", "{b", "{c", GOOD])
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert not a.ok and a.attempts == 3 and len(a.history) == 3
    assert a.raw_text == "{c"
    assert len(api.script) == 1                 # 4th scripted response never consumed


def test_non_retryable_fails_immediately(scripted_api):
    api = scripted_api([ValueError("bad key"), GOOD])
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert not a.ok and a.attempts == 1
    assert a.history == [{"kind": "fatal", "error": "bad key"}]


def test_usage_and_latency_accumulate(scripted_api, monkeypatch):
    import itertools
    api = scripted_api(["{a", GOOD])
    ticks = itertools.chain([100.0], itertools.repeat(100.25))
    monkeypatch.setattr(pu.time, "monotonic", lambda: next(ticks))
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=lambda s: None)
    assert a.usage == {"input_tokens": 20, "output_tokens": 10}
    assert a.latency_ms == 250


def test_backoff_sequence(scripted_api):
    api = scripted_api([RateLimitError("1"), RateLimitError("2"), GOOD])
    sleeps = []
    a = call_with_retry(MESSAGES, 10, parse, call_api_fn=api, sleep_fn=sleeps.append)
    assert a.ok and sleeps == [1, 4]


def test_retry_with_feedback_seeds_conversation(scripted_api):
    api = scripted_api([GOOD])
    a = retry_with_feedback(MESSAGES, "{prior", "second line is wrong", 10, parse,
                            call_api_fn=api, sleep_fn=lambda s: None)
    assert a.ok
    sent = api.calls[0]
    assert sent[1]["content"][0]["text"] == "{prior"
    assert sent[2]["content"][0]["text"] == "second line is wrong"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_retry.py -v`
Expected: FAIL with `ImportError: cannot import name 'call_with_retry'`

- [ ] **Step 3: Add the retry loop to `pipeline_utils.py`**

Add `import time` at the top. Append:

```python
# --- Retry -----------------------------------------------------------------

class ParseError(Exception):
    """Model output is not valid JSON."""


class StructureError(Exception):
    """Model output parsed but is missing or mistyping a required field."""


@dataclass
class Attempt:
    ok: bool
    parsed: Optional[dict]
    attempts: int
    history: list
    error: Optional[str]
    usage: dict
    latency_ms: int
    raw_text: Optional[str]


REPAIR_TEMPLATE = (
    "The previous response could not be used: {error}. "
    "Return the same transcription in the required JSON structure exactly as specified. "
    "Do not change, correct, reorder, or omit any transcribed text."
)
BACKOFF_SECONDS = (1, 4, 10)


def _text_message(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}


def call_with_retry(messages, max_tokens, parse, max_calls=3, call_api_fn=None, sleep_fn=time.sleep):
    """Call the model until parse() accepts the output or max_calls is spent.

    Transport errors are re-sent unchanged after backoff. Parse/structure errors
    continue the conversation with the bad output and a repair instruction.
    Any other exception ends the attempt immediately.
    """
    call_api_fn = call_api_fn or call_api
    conversation = list(messages)
    history = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    started = time.monotonic()
    attempts = 0
    raw = None

    while attempts < max_calls:
        attempts += 1
        try:
            raw, call_usage = call_api_fn(conversation, max_tokens)
        except Exception as exc:
            if is_transient(exc):
                history.append({"kind": "transport", "error": str(exc)})
                logger.warning(f"Transient API error on attempt {attempts}: {exc}")
                if attempts < max_calls:
                    sleep_fn(BACKOFF_SECONDS[min(attempts - 1, len(BACKOFF_SECONDS) - 1)])
                continue
            history.append({"kind": "fatal", "error": str(exc)})
            logger.error(f"Non-retryable API error: {exc}")
            break

        usage["input_tokens"] += call_usage.get("input_tokens", 0)
        usage["output_tokens"] += call_usage.get("output_tokens", 0)

        kind = error = None
        try:
            parsed = parse(raw)
        except ParseError as exc:
            kind, error = "parse", str(exc)
        except StructureError as exc:
            kind, error = "structure", str(exc)

        if error is None:
            return Attempt(True, parsed, attempts, history, None, usage,
                           int((time.monotonic() - started) * 1000), raw)

        history.append({"kind": kind, "error": error})
        logger.warning(f"Unusable model output on attempt {attempts} ({kind}): {error}")
        conversation = conversation + [
            _text_message("assistant", raw),
            _text_message("user", REPAIR_TEMPLATE.format(error=error)),
        ]

    last_error = history[-1]["error"] if history else "no response"
    return Attempt(False, None, attempts, history, last_error, usage,
                   int((time.monotonic() - started) * 1000), raw)


def retry_with_feedback(messages, prior_output, feedback, max_tokens, parse, **kwargs):
    """Re-run a call with the prior output and a caller-supplied correction in context."""
    seeded = list(messages) + [
        _text_message("assistant", prior_output),
        _text_message("user", feedback),
    ]
    return call_with_retry(seeded, max_tokens, parse, **kwargs)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_retry.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline_utils.py tests/test_retry.py
git commit -m "add call_with_retry with backoff and repair turns

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Metrics log and atomic JSON writes

**Files:**
- Modify: `pipeline_utils.py`
- Modify: `.gitignore`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces (in `pipeline_utils`):
  - `METRICS_FILE: str` (env `METRICS_FILE`, default `metrics.jsonl`)
  - `atomic_write_json(path: str, data) -> None`
  - `append_metric(row: dict, path: str | None = None) -> None`
  - `metric_row(meta: CallMeta, status: str, error: str | None) -> dict`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'append_metric'`

- [ ] **Step 3: Add the writers to `pipeline_utils.py`**

Add `import tempfile` at the top. Add `METRICS_FILE = os.getenv('METRICS_FILE', 'metrics.jsonl')` next to `INPUT_FOLDER`. Append:

```python
# --- Persistence -----------------------------------------------------------

def atomic_write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def metric_row(meta, status, error):
    row = asdict(meta)
    row["status"] = status
    row["error"] = error
    return row


def append_metric(row, path=None):
    """Append one JSON line. This file is never truncated by the pipeline."""
    path = path or METRICS_FILE
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
```

- [ ] **Step 4: Add to `.gitignore`**

Append under the `# Responses cache` block:

```
responses_current.json
run_status.json
metrics.jsonl
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_metrics.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add pipeline_utils.py tests/test_metrics.py .gitignore
git commit -m "add permanent metrics log and atomic json writer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `process_images.process_group`

**Files:**
- Rewrite: `process_images.py`
- Create: `tests/test_process_images.py`

**Interfaces:**
- Consumes: `pipeline_utils.get_file_order`, `IMAGE_EXTENSIONS`, `INPUT_FOLDER`, `StageResult`, `logger`.
- Produces: `process_images.resize_image(image_path, max_size=(2000, 2000)) -> str` (returns output path; raises on failure), `process_images.process_group(group_name, input_dir=None) -> StageResult`.

- [ ] **Step 1: Write the failing tests**

`tests/test_process_images.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_process_images.py -v`
Expected: FAIL — `process_group` does not exist (the current module also executes script code at import; the failure may surface as a `SystemExit` or `ImportError`).

- [ ] **Step 3: Rewrite `process_images.py`**

```python
import os
import sys
from PIL import Image
from pillow_heif import register_heif_opener

from pipeline_utils import (
    get_file_order, IMAGE_EXTENSIONS, INPUT_FOLDER, StageResult, logger,
)

register_heif_opener()


def resize_image(image_path, max_size=(2000, 2000)):
    """Shrink to max_size and convert to JPEG in place. Returns the output path."""
    logger.info(f"Processing image: {image_path}")
    with Image.open(image_path) as img:
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.load()
        if img.format != "JPEG":
            output_path = os.path.splitext(image_path)[0] + ".jpg"
            logger.info(f"Converting {img.format} to JPEG: {image_path} -> {output_path}")
        else:
            output_path = image_path
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=85)

    if output_path != image_path and os.path.exists(image_path):
        os.remove(image_path)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Failed to save {output_path}")
    logger.info(f"Successfully processed: {output_path}")
    return output_path


def process_group(group_name, input_dir=None):
    folder = os.path.join(input_dir or INPUT_FOLDER, group_name)
    if not os.path.isdir(folder):
        return StageResult(False, f"Group folder not found: {folder}")

    errors = []
    file_order = get_file_order(folder)
    logger.info(f"Resizing {len(file_order)} files in group {group_name}: {file_order}")
    for image_file in file_order:
        image_path = os.path.join(folder, image_file)
        if not (os.path.isfile(image_path) and image_file.lower().endswith(IMAGE_EXTENSIONS)):
            continue
        try:
            resize_image(image_path)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            errors.append(f"{image_file}: {e}")

    if errors:
        return StageResult(False, "; ".join(errors))
    return StageResult(True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = process_group(sys.argv[1])
        if not result.ok:
            logger.error(result.error)
            sys.exit(1)
    else:
        for name in sorted(os.listdir(INPUT_FOLDER)):
            if os.path.isdir(os.path.join(INPUT_FOLDER, name)):
                process_group(name)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_process_images.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the full suite**

Run: `$PY -m pytest -q`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add process_images.py tests/test_process_images.py
git commit -m "make process_images importable with process_group entry point

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `googlevision_translater.process_group`

**Files:**
- Create: `googlevision_translater.py` (content derived from `googlevision-translater.py`; old file deleted in Task 13)
- Create: `tests/test_ocr.py`

**Interfaces:**
- Consumes: `pipeline_utils.get_file_order`, `resolve_image_path`, `IMAGE_EXTENSIONS`, `INPUT_FOLDER`, `StageResult`, `logger`.
- Produces: `googlevision_translater.extract_text_from_image(image_path) -> str`, `googlevision_translater.process_group(group_name, input_dir=None, extract_fn=None) -> StageResult`. Writes `<stem>.txt` next to each image.

- [ ] **Step 1: Write the failing tests**

`tests/test_ocr.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_ocr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'googlevision_translater'`

- [ ] **Step 3: Create `googlevision_translater.py`**

```python
import os
import sys
from google.cloud import vision_v1 as vision

from pipeline_utils import (
    get_file_order, resolve_image_path, IMAGE_EXTENSIONS, INPUT_FOLDER, StageResult, logger,
)


def extract_text_from_image(image_path):
    """Document text detection via Google Vision. Returns the block-joined text."""
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as f:
        content = f.read()
    response = client.annotate_image({
        "image": {"content": content},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
    })
    if response.error.message:
        raise RuntimeError(f"Google Vision API error: {response.error.message}")

    blocks = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            blocks.append(" ".join(
                "".join(symbol.text for symbol in word.symbols)
                for paragraph in block.paragraphs
                for word in paragraph.words
            ))
    text = "\n\n".join(blocks)
    logger.info(f"OCR content ({len(text)} chars): {text[:200]}{'...' if len(text) > 200 else ''}")
    return text


def process_group(group_name, input_dir=None, extract_fn=None):
    extract_fn = extract_fn or extract_text_from_image
    folder = os.path.join(input_dir or INPUT_FOLDER, group_name)
    if not os.path.isdir(folder):
        return StageResult(False, f"Group folder not found: {folder}")

    errors = []
    file_order = get_file_order(folder)
    logger.info(f"OCR for {len(file_order)} files in group {group_name}: {file_order}")
    for image_file in file_order:
        resolved = resolve_image_path(folder, image_file)
        if resolved is None:
            logger.warning(f"Skipping OCR for {image_file} — file not found")
            continue
        name, path = resolved
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        try:
            text = extract_fn(path)
        except Exception as e:
            logger.error(f"OCR failed for {name}: {e}")
            errors.append(f"{name}: {e}")
            continue
        output_path = os.path.join(folder, os.path.splitext(name)[0] + ".txt")
        with open(output_path, "w") as f:
            f.write(text)
        logger.info(f"Text extracted and saved to {output_path}")

    if errors:
        return StageResult(False, "; ".join(errors))
    return StageResult(True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = process_group(sys.argv[1])
        if not result.ok:
            logger.error(result.error)
            sys.exit(1)
    else:
        for name in sorted(os.listdir(INPUT_FOLDER)):
            if os.path.isdir(os.path.join(INPUT_FOLDER, name)):
                process_group(name)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_ocr.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add googlevision_translater.py tests/test_ocr.py
git commit -m "add importable googlevision_translater with process_group

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `note_translater` — prompts, parsers, warnings, group validation

**Files:**
- Create: `note_translater.py` (pure helpers moved verbatim from `gpt4-note-translater.py`; old file deleted in Task 13)
- Create: `tests/test_note_parsers.py`

**Interfaces:**
- Consumes: `pipeline_utils.ParseError`, `StructureError`, `IMAGE_EXTENSIONS`, `logger`; `obsidian_tags.load_saved_tags`.
- Produces (in `note_translater`):
  - `DATE_FORMAT`, `parse_date_string(s) -> str | None`, `clean_json_text(text) -> str`, `generate_uuid(filenames, model) -> str`, `encode_image(path) -> str`, `create_text_path(image_path) -> str` (moved unchanged)
  - `load_obsidian_tags() -> list[str]`, `build_single_prompt(tags) -> str`, `build_multi_prompt(tags) -> str`, `prompt_hash(text) -> str`
  - `parse_page(text) -> dict` (raises `ParseError` / `StructureError`), `parse_summary(text) -> dict`
  - `INLINE_MARKER_RE`, `MIN_TRANSCRIPTION_LENGTH = 20`, `page_warnings(data) -> list[str]`
  - `validate_group(folder_path, file_order, summary) -> list[str]`

- [ ] **Step 1: Write the failing tests**

`tests/test_note_parsers.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_note_parsers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'note_translater'`

- [ ] **Step 3: Create `note_translater.py` (part 1)**

Copy these functions **verbatim** from `gpt4-note-translater.py` into the new file: `parse_date_string` (lines 157–206), `generate_uuid` (209–216), `create_text_path` (219–222), `clean_json_text` (225–297), `encode_image` (145–147). Then add the rest below. The file at the end of this step:

```python
import os
import re
import json
import base64
import hashlib
import unicodedata
from datetime import datetime

from obsidian_tags import load_saved_tags
from pipeline_utils import (
    ParseError, StructureError, IMAGE_EXTENSIONS, logger,
)

DATE_FORMAT = "%Y_%m_%d"
DATE_FORMAT_DISPLAY = "YYYY_MM_DD"
MIN_TRANSCRIPTION_LENGTH = 20
INLINE_MARKER_RE = re.compile(r"\[\?[^\]]*\]")


# ---- verbatim from gpt4-note-translater.py -------------------------------
# parse_date_string, generate_uuid, create_text_path, clean_json_text, encode_image
# (paste here, unchanged)


# ---- prompts -------------------------------------------------------------

single_image_prompt = '''Here are your directions:
I am sending you a single handwritten note please transcribe it accurately.
This image is accompanied by extracted text from Google Vision API with the same filename.
Use the image and text to assist in transcribing the words accurately. If you see a word crossed out in the image, ignore it.
Output text in markdown format and wrap it in JSON, using the included template below for structure. Your transcription should replace the <transcription_here> in the template.
Do not include ```markdown code tags or ```json code tags, otherwise structure using normal JSON containing normal markdown syntax. (DO NOT WRAP in markdown or json code blocks).
I want you to devise a simple title, based on the content of the note, and insert it into the <title_here> field.

CRITICAL - Whitespace Rules:
- Only add line breaks or whitespace where they actually appear in the original handwritten text
- DO NOT add extra paragraph breaks, blank lines, or spacing that isn't in the source material
- Transcribe the text with the exact spacing and flow as written

CRITICAL - Date Handling:
- If a date appears in the document, it must appear in TWO places:
  1. In the <date_here> metadata field (for use in filenames)
  2. In the <transcription_here> text at its original location (preserve it in the transcription)
- Never guess years on a date, include only what is explicitly stated in the document.
- Dates should NOT be removed from the transcription text when extracting them to metadata.

Uncertain words:
- In the "uncertain" array, list any word or short phrase you could not read with confidence. Give the text exactly as you transcribed it and a few surrounding words as context.
- Put your best reading in the transcription itself. Do not mark, bracket, or annotate uncertain words inside the transcription text.
- Use an empty array if every word was clear.
'''

multi_prompt = '''You are analyzing handwritten notes from sequential pages of the same document.

CRITICAL INSTRUCTIONS:
1. **Continuous Transcription**: Transcribe ALL pages as ONE flowing document
   - Maintain natural continuity across page boundaries
   - If a sentence continues from one page to the next, join them naturally
   - DO NOT add page numbers or separators between pages
   - Focus on creating readable, continuous prose while sticking to the original text precisely
   - If a word is crossed out in the image, ignore it.

   **CRITICAL - Whitespace Rules**:
   - Only add line breaks or whitespace where they actually appear in the original handwritten text
   - DO NOT add extra paragraph breaks, blank lines, or spacing that isn't in the source material
   - Transcribe the text with the exact spacing and flow as written
   - Ignore page boundaries - if text flows continuously, keep it continuous

2. **Summary Metadata**: After transcribing, provide:
   - Overall title for the document
   - Date (or date range if multiple dates appear)
   - Summary of the main themes/topics
   - Relevant tags

3. **CRITICAL - Date Handling**:
   Dates must appear in TWO places:
   - IN THE CONTINUOUS TRANSCRIPTION: Keep all dates at their original locations in the text exactly as they appear
   - IN THE METADATA DATE FIELD: List the primary date (or all dates if multiple exist across pages)

   A date value listed by itself in the manner of "dating" a document should be the primary data point for the metadata date field.
   DO NOT remove dates from the continuous transcription when extracting them to metadata - they must remain in both places.

Do not include ```markdown code tags or ```json code tags, otherwise structure using normal JSON containing normal markdown syntax. (DO NOT WRAP in markdown or json code blocks).
The images and OCR text are provided below in order.
'''

expected_format_single_prompt = '''
{
  "title": "<title_here>",
  "date": "<date_here>",
  "transcription": "<transcription_here>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"],
  "uncertain": [{"text": "<word or phrase as transcribed>", "context": "<a few surrounding words>"}]
}
'''

expected_format_multiprompt = '''
{
  "title": "<title_here>",
  "date": "<date_here>",
  "summary": "<summary_here>",
  "continuous_transcription": "<full_continuous_text>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
}
'''

date_format_rules = f'''
Date formatting rules (METADATA ONLY - keep original format in transcription):
1. In the metadata date field, format dates as {DATE_FORMAT_DISPLAY}, such as 2025_08_01.
2. In the transcription text, keep dates in their original format exactly as they appear in the document.
3. Do not include / in metadata dates or any characters that would disrupt their use as a filename.
'''


def _tag_guidance(tags):
    if not tags:
        return ""
    return f'''
When choosing tags, consider using tags from this existing vocabulary when appropriate:
{", ".join(tags)}

CRITICAL - Tag Limit:
- Provide EXACTLY 3 tags maximum - choose the 3 most relevant tags only
- If fewer than 3 tags are appropriate, provide fewer
- DO NOT exceed 3 tags under any circumstances'''


def load_obsidian_tags():
    return load_saved_tags(os.getenv('OBSIDIAN_TAGS_FILE', 'obsidian_tags.json'))


def build_single_prompt(tags):
    return single_image_prompt + date_format_rules + expected_format_single_prompt + _tag_guidance(tags)


def build_multi_prompt(tags):
    return multi_prompt + date_format_rules + expected_format_multiprompt + _tag_guidance(tags)


def prompt_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---- parsers -------------------------------------------------------------

def _load_object(text):
    try:
        data = json.loads(clean_json_text(text))
    except json.JSONDecodeError as e:
        raise ParseError(str(e))
    if not isinstance(data, dict):
        raise StructureError(f"expected a JSON object, got {type(data).__name__}")
    return data


def _require(data, field, kind):
    if field not in data:
        raise StructureError(f"missing field: {field}")
    if not isinstance(data[field], kind):
        raise StructureError(f"field {field} must be {kind.__name__}")


def parse_page(text):
    data = _load_object(text)
    _require(data, "title", str)
    _require(data, "date", str)
    _require(data, "transcription", str)
    _require(data, "tags", list)
    uncertain = data.get("uncertain", [])
    if not isinstance(uncertain, list) or not all(
        isinstance(u, dict) and isinstance(u.get("text"), str) for u in uncertain
    ):
        raise StructureError("field uncertain must be a list of {text, context} objects")
    data["uncertain"] = [
        {"text": u["text"], "context": str(u.get("context", ""))} for u in uncertain
    ]
    return data


def parse_summary(text):
    data = _load_object(text)
    _require(data, "title", str)
    _require(data, "date", str)
    _require(data, "summary", str)
    _require(data, "continuous_transcription", str)
    _require(data, "tags", list)
    return data


# ---- warnings (never retried) --------------------------------------------

def page_warnings(data):
    warnings = []
    text = data.get("transcription", "")
    if INLINE_MARKER_RE.search(text):
        warnings.append("inline uncertainty marker in transcription")
    if len(text.strip()) < MIN_TRANSCRIPTION_LENGTH:
        warnings.append(
            f"Suspiciously short transcription ({len(text.strip())} chars) — may be a failed page")
    tags = data.get("tags", [])
    if len(tags) > 3:
        warnings.append(f"Too many tags ({len(tags)}) — expected 3 max: {tags}")
    return warnings


def validate_group(folder_path, file_order, summary):
    """Order-vs-disk checks plus summary sanity. Returns warning strings."""
    warnings = []

    def stem_to_jpg(filename):
        return os.path.splitext(filename)[0] + '.jpg'

    on_disk = {stem_to_jpg(f) for f in os.listdir(folder_path)
               if f.lower().endswith(IMAGE_EXTENSIONS)}
    in_order = {stem_to_jpg(f) for f in file_order}
    if not os.path.exists(os.path.join(folder_path, 'order.json')):
        warnings.append("No order.json found — using alphabetical sort. Pages may be out of order.")
    if on_disk - in_order:
        warnings.append(f"Files on disk not in order list (will be skipped): {sorted(on_disk - in_order)}")
    if in_order - on_disk:
        warnings.append(f"Files in order list not found on disk (will be missing): {sorted(in_order - on_disk)}")

    if summary:
        if len(summary.get('tags', [])) > 3:
            warnings.append(f"Summary response returned {len(summary['tags'])} tags — expected 3 max: {summary['tags']}")
        continuous = summary.get('continuous_transcription', '')
        if len(continuous.strip()) < MIN_TRANSCRIPTION_LENGTH:
            warnings.append(f"Continuous transcription in summary is suspiciously short ({len(continuous.strip())} chars)")

    for w in warnings:
        logger.warning(f"[VALIDATION] {w}")
    return warnings
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_note_parsers.py -v`
Expected: 25 passed

- [ ] **Step 5: Commit**

```bash
git add note_translater.py tests/test_note_parsers.py
git commit -m "add note_translater prompts, parsers and validators with uncertain field

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: `note_translater.process_group` and persistence

**Files:**
- Modify: `note_translater.py`
- Create: `tests/test_note_translater.py`

**Interfaces:**
- Consumes: Task 8 helpers; `pipeline_utils.call_with_retry`, `get_file_order`, `resolve_image_path`, `INPUT_FOLDER`, `get_provider`, `get_model`, `CallMeta`, `PageResult`, `GroupResult`, `rollup_status`, `append_metric`, `metric_row`, `atomic_write_json`.
- Produces:
  - `note_translater.CURRENT_RESPONSES_FILE`, `RESPONSES_FILE` (from env, defaults `responses_current.json` / `responses.json`)
  - `build_page_messages(prompt, pair) -> list[dict]`, `build_summary_messages(prompt, pairs) -> list[dict]` where `pair = {"filename", "path", "base64", "ocr_text"}`
  - `process_group(group_name, on_update=None, input_dir=None, call_api_fn=None, run_id=None, sleep_fn=time.sleep) -> GroupResult`
  - `on_update` is called as `on_update(group_name, **fields)` with any of `stage`, `pages_total`, `pages_done`, `attempts`, `warnings`, `status`, `error`.
  - `save_group_result(result: GroupResult, current_file=None, history_file=None) -> str` (returns the uuid key)

- [ ] **Step 1: Write the failing tests**

`tests/test_note_translater.py`:

```python
import json
import os
import pytest
from conftest import PAGE_OK, SUMMARY_OK

import note_translater as nt


class RateLimitError(Exception):
    pass


def _updates_recorder():
    calls = []

    def on_update(group_name, **fields):
        calls.append((group_name, fields))
    return calls, on_update


def _run(group_folder, script, **kw):
    from conftest import ScriptedApi
    api = ScriptedApi(script)
    calls, on_update = _updates_recorder()
    result = nt.process_group(
        "n1", on_update=on_update, input_dir=str(group_folder.parent),
        call_api_fn=api, run_id="r1", sleep_fn=lambda s: None, **kw)
    return result, api, calls


def test_two_pages_all_ok(group_folder):
    result, api, calls = _run(group_folder, [json.dumps(PAGE_OK), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])
    assert result.status == "done"
    assert [p.status for p in result.pages] == ["done", "done"]
    assert result.summary["title"] == "Two page note"
    assert result.summary_attempts == 1
    assert result.pages[0].meta.kind == "page" and result.summary_meta.kind == "summary"
    assert result.pages[0].meta.prompt_hash != result.summary_meta.prompt_hash
    assert len(api.calls) == 3
    # first page call carries prompt, OCR text and image in that order
    blocks = api.calls[0][0]["content"]
    assert blocks[0]["type"] == "text" and "uncertain" in blocks[0]["text"]
    assert blocks[1] == {"type": "text", "text": "ocr text one"}
    assert blocks[2]["type"] == "image"


def test_on_update_sequence(group_folder):
    _, _, calls = _run(group_folder, [json.dumps(PAGE_OK), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])
    stages = [f.get("stage") for _, f in calls if "stage" in f]
    assert stages == ["transcribing", "summarising", "complete"]
    done_counts = [f["pages_done"] for _, f in calls if "pages_done" in f]
    assert done_counts == [0, 1, 2]
    assert calls[-1][1]["status"] == "done"


def test_failed_page_keeps_sibling_and_fails_group(group_folder):
    result, api, calls = _run(group_folder, [json.dumps(PAGE_OK), "{a", "{b", "{c", json.dumps(SUMMARY_OK)])
    assert result.status == "failed"
    assert result.pages[0].status == "done" and result.pages[0].data["title"] == "Grocery list"
    assert result.pages[1].status == "failed" and result.pages[1].attempts == 3
    assert "page2.jpg" in result.errors[0] and "after 3 attempts" in result.errors[0]
    assert calls[-1][1]["status"] == "failed" and "page2.jpg" in calls[-1][1]["error"]
    # summary still attempted so the retry surface is complete
    assert result.summary is not None


def test_summary_failure_fails_group(group_folder):
    result, *_ = _run(group_folder, [json.dumps(PAGE_OK), json.dumps(PAGE_OK), RateLimitError("x"), RateLimitError("y"), RateLimitError("z")])
    assert result.status == "failed" and result.summary is None
    assert result.summary_attempts == 3 and "summary" in result.errors[0]


def test_warning_page_gives_warning_group(group_folder):
    short = dict(PAGE_OK, transcription="hi")
    result, *_ = _run(group_folder, [json.dumps(short), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])
    assert result.pages[0].status == "warning" and result.status == "warning"


def test_single_page_skips_summary(group_folder):
    (group_folder / "order.json").write_text(json.dumps({"files": ["page1.jpg"]}))
    result, api, _ = _run(group_folder, [json.dumps(PAGE_OK)])
    assert len(api.calls) == 1 and result.summary is None
    assert result.status == "warning"     # page2.jpg on disk but not in order -> group warning


def test_empty_ocr_text_block_is_omitted(group_folder):
    (group_folder / "page1.txt").write_text("")
    _, api, _ = _run(group_folder, [json.dumps(PAGE_OK), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])
    types = [b["type"] for b in api.calls[0][0]["content"]]
    assert types == ["text", "image"]


def test_no_images_is_failed(tmp_path):
    (tmp_path / "n1").mkdir()
    result = nt.process_group("n1", input_dir=str(tmp_path), call_api_fn=lambda m, t: ("", {}), run_id="r1")
    assert result.status == "failed" and "no images" in result.errors[0]


def test_persists_current_and_history_and_metrics(group_folder, tmp_path, monkeypatch):
    current = tmp_path / "cur.json"
    history = tmp_path / "hist.json"
    metrics = tmp_path / "m.jsonl"
    monkeypatch.setattr(nt, "CURRENT_RESPONSES_FILE", str(current))
    monkeypatch.setattr(nt, "RESPONSES_FILE", str(history))
    monkeypatch.setattr("pipeline_utils.METRICS_FILE", str(metrics))
    _run(group_folder, [json.dumps(PAGE_OK), "{a", json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)])
    cur = json.loads(current.read_text())
    (key, group), = cur.items()
    assert group["status"] == "done" and group["pages"][1]["attempts"] == 2
    assert key in json.loads(history.read_text())
    rows = [json.loads(l) for l in metrics.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["page", "page", "summary"]
    assert rows[1]["attempts"] == 2 and rows[1]["input_tokens"] == 20


def test_history_key_dedups_on_rerun(group_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(nt, "CURRENT_RESPONSES_FILE", str(tmp_path / "cur.json"))
    monkeypatch.setattr(nt, "RESPONSES_FILE", str(tmp_path / "hist.json"))
    script = [json.dumps(PAGE_OK), json.dumps(PAGE_OK), json.dumps(SUMMARY_OK)]
    _run(group_folder, list(script))
    _run(group_folder, list(script))
    assert len(json.loads((tmp_path / "hist.json").read_text())) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_note_translater.py -v`
Expected: FAIL with `AttributeError: module 'note_translater' has no attribute 'process_group'`

- [ ] **Step 3: Add `process_group` and persistence to `note_translater.py`**

Extend the imports at the top of `note_translater.py`:

```python
import time
from pipeline_utils import (
    ParseError, StructureError, IMAGE_EXTENSIONS, INPUT_FOLDER, logger,
    get_file_order, resolve_image_path, get_provider, get_model,
    call_with_retry, CallMeta, PageResult, GroupResult, rollup_status,
    append_metric, metric_row, atomic_write_json,
)

RESPONSES_FILE = os.getenv('RESPONSES_FILE', 'responses.json')
CURRENT_RESPONSES_FILE = os.getenv('CURRENT_RESPONSES_FILE', 'responses_current.json')
PAGE_MAX_TOKENS = 10000
SUMMARY_MAX_TOKENS = 64000
```

Append to the end of the file:

```python
# ---- message building ----------------------------------------------------

def _load_pairs(folder_path, file_order):
    pairs = []
    for image_file in file_order:
        resolved = resolve_image_path(folder_path, image_file)
        if resolved is None:
            logger.warning(f"Skipping {image_file} — file not found and no .jpg equivalent exists")
            continue
        name, path = resolved
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        ocr_text = ""
        text_path = create_text_path(path)
        if os.path.exists(text_path):
            with open(text_path, "r") as f:
                ocr_text = f.read()
        pairs.append({"filename": name, "path": path,
                      "base64": encode_image(path), "ocr_text": ocr_text})
    return pairs


def build_page_messages(prompt, pair):
    content = [{"type": "text", "text": prompt}]
    if pair["ocr_text"]:                      # Anthropic rejects empty text blocks
        content.append({"type": "text", "text": pair["ocr_text"]})
    content.append({"type": "image", "base64": pair["base64"]})
    return [{"role": "user", "content": content}]


def build_summary_messages(prompt, pairs):
    content = [{"type": "text", "text": prompt}]
    for pair in pairs:
        content.append({"type": "image", "base64": pair["base64"]})
        if pair["ocr_text"]:
            content.append({"type": "text", "text": f"\n\nOCR text:\n{pair['ocr_text']}\n"})
    return [{"role": "user", "content": content}]


# ---- group processing ----------------------------------------------------

def _meta(run_id, group_name, filename, kind, phash, attempt, text, uncertain_count):
    return CallMeta(
        run_id=run_id, group_name=group_name, filename=filename, kind=kind,
        provider=get_provider(), model=get_model(), prompt_hash=phash,
        attempts=attempt.attempts, latency_ms=attempt.latency_ms,
        input_tokens=attempt.usage.get("input_tokens", 0),
        output_tokens=attempt.usage.get("output_tokens", 0),
        word_count=len(text.split()), uncertain_count=uncertain_count,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


def process_group(group_name, on_update=None, input_dir=None, call_api_fn=None,
                  run_id=None, sleep_fn=time.sleep):
    on_update = on_update or (lambda name, **fields: None)
    run_id = run_id or datetime.now().isoformat(timespec="seconds")
    folder_path = os.path.join(input_dir or INPUT_FOLDER, group_name)
    retry_kwargs = {"call_api_fn": call_api_fn, "sleep_fn": sleep_fn}

    file_order = get_file_order(folder_path) if os.path.isdir(folder_path) else []
    pairs = _load_pairs(folder_path, file_order) if file_order else []
    logger.info(f"Transcribing group {group_name}: {[p['filename'] for p in pairs]}")

    if not pairs:
        result = GroupResult(group_name=group_name, status="failed", file_order=file_order,
                             image_paths=[], pages=[], errors=["no images found in group"])
        on_update(group_name, stage="complete", status="failed", error=result.errors[0])
        return result

    tags = load_obsidian_tags()
    single_prompt = build_single_prompt(tags)
    multi_prompt = build_multi_prompt(tags)
    single_hash, multi_hash = prompt_hash(single_prompt), prompt_hash(multi_prompt)

    pages, errors, total_attempts = [], [], 0
    on_update(group_name, stage="transcribing", pages_total=len(pairs), pages_done=0, attempts=0)

    for pair in pairs:
        attempt = call_with_retry(build_page_messages(single_prompt, pair),
                                  PAGE_MAX_TOKENS, parse_page, **retry_kwargs)
        total_attempts += attempt.attempts
        if attempt.ok:
            data = attempt.parsed
            warnings = page_warnings(data)
            page = PageResult(
                filename=pair["filename"], status="warning" if warnings else "done",
                attempts=attempt.attempts, data=data, uncertain=data["uncertain"],
                warnings=warnings, history=attempt.history,
                meta=_meta(run_id, group_name, pair["filename"], "page", single_hash,
                           attempt, data["transcription"], len(data["uncertain"])))
        else:
            error = f"{pair['filename']}: {attempt.error} after {attempt.attempts} attempts"
            errors.append(f"page {error}")
            page = PageResult(
                filename=pair["filename"], status="failed", attempts=attempt.attempts,
                error=attempt.error, history=attempt.history,
                meta=_meta(run_id, group_name, pair["filename"], "page", single_hash,
                           attempt, attempt.raw_text or "", 0))
        append_metric(metric_row(page.meta, page.status, page.error))
        pages.append(page)
        on_update(group_name, pages_done=len(pages), attempts=total_attempts,
                  warnings=[w for p in pages for w in p.warnings])

    summary = summary_meta = summary_error = None
    summary_attempts = 0
    if len(pairs) > 1:
        on_update(group_name, stage="summarising")
        attempt = call_with_retry(build_summary_messages(multi_prompt, pairs),
                                  SUMMARY_MAX_TOKENS, parse_summary, **retry_kwargs)
        summary_attempts = attempt.attempts
        total_attempts += attempt.attempts
        if attempt.ok:
            summary = attempt.parsed
            text = summary["continuous_transcription"]
        else:
            summary_error = f"{attempt.error} after {attempt.attempts} attempts"
            errors.append(f"summary: {summary_error}")
            text = attempt.raw_text or ""
        summary_meta = _meta(run_id, group_name, None, "summary", multi_hash, attempt, text, 0)
        append_metric(metric_row(summary_meta, "failed" if summary_error else "done", summary_error))

    group_warnings = validate_group(folder_path, file_order, summary)
    status = rollup_status([p.status for p in pages], summary_error, group_warnings)
    result = GroupResult(
        group_name=group_name, status=status, file_order=file_order,
        image_paths=[p["path"] for p in pairs], pages=pages,
        summary=summary, summary_attempts=summary_attempts, summary_meta=summary_meta,
        summary_error=summary_error, warnings=group_warnings, errors=errors,
    )
    save_group_result(result)
    on_update(group_name, stage="complete", status=status, attempts=total_attempts,
              warnings=group_warnings + [w for p in pages for w in p.warnings],
              error="; ".join(errors) if errors else None)
    logger.info(f"Group {group_name} finished with status {status}")
    return result


# ---- persistence ---------------------------------------------------------

def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _unique_key(history, uuid):
    if uuid not in history:
        return uuid
    n = 1
    while f"{uuid}_{n}" in history:
        n += 1
    return f"{uuid}_{n}"


def save_group_result(result, current_file=None, history_file=None):
    """Write the group into the per-run file (bare uuid key) and append to history."""
    current_file = current_file or CURRENT_RESPONSES_FILE
    history_file = history_file or RESPONSES_FILE
    uuid = generate_uuid([p.filename for p in result.pages] or result.file_order or [result.group_name],
                         get_model())
    payload = result.to_dict()

    current = _load_json(current_file)
    current[uuid] = payload
    atomic_write_json(current_file, current)

    history = _load_json(history_file)
    history[_unique_key(history, uuid)] = payload
    atomic_write_json(history_file, history)
    return uuid


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        names = [sys.argv[1]]
    else:
        names = sorted(n for n in os.listdir(INPUT_FOLDER)
                       if os.path.isdir(os.path.join(INPUT_FOLDER, n)))
    exit_code = 0
    for name in names:
        if process_group(name).status == "failed":
            exit_code = 1
    sys.exit(exit_code)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_note_translater.py -v`
Expected: 10 passed

- [ ] **Step 5: Run the full suite**

Run: `$PY -m pytest -q`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add note_translater.py tests/test_note_translater.py
git commit -m "add note_translater.process_group with retry, page results and metrics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: `export_responses.export_run`

**Files:**
- Rewrite: `export_responses.py`
- Create: `tests/test_export.py`

**Interfaces:**
- Consumes: `GroupResult.to_dict()` shape (a dict with `group_name`, `status`, `image_paths`, `pages[].data`, `summary`).
- Produces: `export_responses.sanitize_filename(s) -> str` (unchanged), `build_markdown(group: dict) -> tuple[str, str]` (folder name, markdown), `export_run(results: list[dict], output_dir: str) -> list[dict]` each `{"group_name": str, "ok": bool, "path": str | None, "error": str | None}`. Skips `failed` groups (they are not in the returned list).

- [ ] **Step 1: Write the failing tests**

`tests/test_export.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_export.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_markdown'` (the current module also runs its export at import time).

- [ ] **Step 3: Rewrite `export_responses.py`**

```python
import os
import re
import sys
import json
import shutil

from pipeline_utils import logger

OUTPUT_FOLDER = os.path.expanduser(os.getenv('OUTPUT_FOLDER', './markdown_output'))
CURRENT_RESPONSES_FILE = os.getenv('CURRENT_RESPONSES_FILE', 'responses_current.json')
EXPORTABLE = ("done", "warning")


def sanitize_filename(filename):
    if not filename:
        return 'Untitled'
    sanitized = re.sub(r'[/\\:*?"<>|]', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_ ')
    return sanitized or 'Untitled'


def build_markdown(group):
    """(folder_name, markdown) for one exportable group dict."""
    summary = group.get("summary")
    pages = group.get("pages", [])
    if summary:
        title, date, tags = summary.get("title", ""), summary.get("date", ""), summary.get("tags", [])
    elif pages and pages[0].get("data"):
        data = pages[0]["data"]
        title, date, tags = data.get("title", ""), data.get("date", ""), data.get("tags", [])
    else:
        raise ValueError(f"group {group.get('group_name')} has no summary and no page data")

    folder_name = f"{sanitize_filename(date) if date else 'Unknown_Date'} - {sanitize_filename(title)}"

    md = f"# {title}\n\n"
    if summary:
        md += f"## Summary\n\n{summary.get('summary', '')}\n\n"
    md += f"**Date:** {date}\n\n"
    if tags:
        md += f"**Tags:** {' '.join(f'#{t}' for t in tags)}\n\n"

    if summary and summary.get("continuous_transcription"):
        md += f"{summary['continuous_transcription']}\n\n"
    else:
        for page in pages:
            data = page.get("data") or {}
            if summary and data.get("date"):
                md += f"{data['date']}\n\n"
            md += f"{data.get('transcription', '')}\n\n"
    return folder_name, md


def _export_group(group, output_dir):
    folder_name, md = build_markdown(group)
    candidate, counter = folder_name, 2
    while os.path.exists(os.path.join(output_dir, candidate)):
        candidate = f"{folder_name}_{counter}"
        counter += 1
    folder_path = os.path.join(output_dir, candidate)
    images_path = os.path.join(folder_path, "images")
    os.makedirs(images_path, exist_ok=True)

    with open(os.path.join(folder_path, f"{candidate}.md"), "w") as f:
        f.write(md)
    for image_path in group.get("image_paths", []):
        if os.path.exists(image_path):
            shutil.copy2(image_path, images_path)
        else:
            logger.warning(f"Image path does not exist: {image_path}")
    return folder_path


def export_run(results, output_dir):
    """Export done/warning groups. Returns one status dict per attempted group."""
    os.makedirs(output_dir, exist_ok=True)
    report = []
    for group in results:
        name = group.get("group_name")
        if group.get("status") not in EXPORTABLE:
            logger.info(f"Skipping export for {name} (status {group.get('status')})")
            continue
        try:
            path = _export_group(group, output_dir)
            logger.info(f"Exported {name} to {path}")
            report.append({"group_name": name, "ok": True, "path": path, "error": None})
        except Exception as e:
            logger.error(f"Export failed for {name}: {e}")
            report.append({"group_name": name, "ok": False, "path": None, "error": str(e)})
    return report


if __name__ == "__main__":
    if not os.path.exists(CURRENT_RESPONSES_FILE):
        logger.warning(f"No {CURRENT_RESPONSES_FILE} found — nothing to export.")
        sys.exit(0)
    with open(CURRENT_RESPONSES_FILE) as f:
        groups = list(json.load(f).values())
    report = export_run(groups, OUTPUT_FOLDER)
    sys.exit(0 if all(r["ok"] for r in report) else 1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_export.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add export_responses.py tests/test_export.py
git commit -m "make export_responses importable; export only done/warning groups

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: `app.py` in-process orchestration and status contract

**Files:**
- Modify: `app.py` (imports lines 1–12, progress globals lines 29–44, `process_images` route lines 461–645, `get_progress` lines 675–678)
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes: `process_images.process_group`, `googlevision_translater.process_group`, `note_translater.process_group`, `export_responses.export_run`, `pipeline_utils.atomic_write_json`, `pipeline_utils.INPUT_FOLDER`.
- Produces:
  - `app.PIPELINE: dict` with keys `resize`, `ocr`, `transcribe`, `export` (monkeypatch point for tests)
  - `app.RUN_STATUS_FILE`, `app.CURRENT_RESPONSES_FILE`
  - `GET /api/progress` payload per spec Section 3 (`status`, `run_id`, `total_groups`, `current_group`, `completed_groups`, `percentage`, `current_step`, `groups[]` with `name`, `label`, `status`, `stage`, `pages_done`, `pages_total`, `attempts`, `warnings`, `error`)
  - `POST /api/process` response unchanged in shape: `{message, groups_processed, total_groups, failed_groups}` with 200 / 207 / 500.

- [ ] **Step 1: Write the failing tests**

`tests/test_app.py`:

```python
import json
import os
import pytest
from conftest import make_jpeg

import app as app_module
from pipeline_utils import StageResult, GroupResult, PageResult


@pytest.fixture
def client(tmp_path, monkeypatch):
    temp = tmp_path / "temp"
    inp = tmp_path / "input"
    temp.mkdir()
    monkeypatch.setattr(app_module, "TEMP_FOLDER", str(temp))
    monkeypatch.setattr(app_module, "INPUT_FOLDER", str(inp))
    monkeypatch.setattr(app_module, "OUTPUT_FOLDER", str(tmp_path / "out"))
    monkeypatch.setattr(app_module, "RUN_STATUS_FILE", str(tmp_path / "run_status.json"))
    monkeypatch.setattr(app_module, "CURRENT_RESPONSES_FILE", str(tmp_path / "cur.json"))
    app_module.processing_progress.clear()
    app_module.processing_progress.update(app_module._new_progress())
    make_jpeg(temp / "a.jpg")
    make_jpeg(temp / "b.jpg")
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _ok_stage(group_name, input_dir=None):
    return StageResult(True)


def _transcribe_factory(fail_groups=()):
    def transcribe(group_name, on_update=None, input_dir=None, run_id=None):
        on_update(group_name, stage="transcribing", pages_total=1, pages_done=0, attempts=0)
        status = "failed" if group_name in fail_groups else "done"
        on_update(group_name, stage="complete", status=status, pages_done=1, attempts=2,
                  error="page a.jpg: boom after 3 attempts" if status == "failed" else None)
        return GroupResult(group_name=group_name, status=status, file_order=["a.jpg"],
                           image_paths=[], pages=[PageResult(filename="a.jpg", status=status)],
                           errors=["page a.jpg: boom after 3 attempts"] if status == "failed" else [])
    return transcribe


def _export(results, output_dir):
    return [{"group_name": r["group_name"], "ok": True, "path": "/x", "error": None} for r in results]


def _post(client, groups):
    return client.post("/api/process", data=json.dumps({"groups": groups}),
                       content_type="application/json")


def test_all_groups_succeed(client, monkeypatch):
    exported = []
    monkeypatch.setitem(app_module.PIPELINE, "resize", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "ocr", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", _transcribe_factory())
    monkeypatch.setitem(app_module.PIPELINE, "export", lambda r, o: exported.extend(r) or _export(r, o))
    resp = _post(client, [{"images": ["a.jpg"]}, {"images": ["b.jpg"]}])
    assert resp.status_code == 200
    progress = client.get("/api/progress").get_json()
    assert progress["status"] == "completed" and progress["percentage"] == 100
    assert [g["status"] for g in progress["groups"]] == ["done", "done"]
    assert progress["groups"][0]["label"] == "1 page"
    assert [r["group_name"] for r in exported] == ["n1", "n2"]
    assert os.path.exists(app_module.RUN_STATUS_FILE)


def test_failed_group_does_not_stop_run(client, monkeypatch):
    exported = []
    monkeypatch.setitem(app_module.PIPELINE, "resize", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "ocr", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", _transcribe_factory(fail_groups=("n1",)))
    monkeypatch.setitem(app_module.PIPELINE, "export", lambda r, o: exported.extend(r) or _export(r, o))
    resp = _post(client, [{"images": ["a.jpg"]}, {"images": ["b.jpg"]}])
    assert resp.status_code == 207
    body = resp.get_json()
    assert body["groups_processed"] == 1 and body["failed_groups"] == ["n1"]
    groups = client.get("/api/progress").get_json()["groups"]
    assert groups[0]["status"] == "failed" and "boom" in groups[0]["error"]
    assert groups[1]["status"] == "done"
    assert [r["group_name"] for r in exported] == ["n2"]


def test_stage_failure_marks_group_failed(client, monkeypatch):
    monkeypatch.setitem(app_module.PIPELINE, "resize", lambda g, input_dir=None: StageResult(False, "corrupt"))
    monkeypatch.setitem(app_module.PIPELINE, "ocr", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", _transcribe_factory())
    monkeypatch.setitem(app_module.PIPELINE, "export", _export)
    resp = _post(client, [{"images": ["a.jpg"]}])
    assert resp.status_code == 500
    g = client.get("/api/progress").get_json()["groups"][0]
    assert g["status"] == "failed" and g["stage"] == "resizing" and "corrupt" in g["error"]


def test_unexpected_exception_is_isolated(client, monkeypatch):
    def explode(group_name, input_dir=None):
        raise RuntimeError("kaboom")
    monkeypatch.setitem(app_module.PIPELINE, "resize", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "ocr", explode)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", _transcribe_factory())
    monkeypatch.setitem(app_module.PIPELINE, "export", _export)
    resp = _post(client, [{"images": ["a.jpg"]}, {"images": ["b.jpg"]}])
    assert resp.status_code == 207
    groups = client.get("/api/progress").get_json()["groups"]
    assert "kaboom" in groups[0]["error"] and groups[1]["status"] == "done"


def test_progress_falls_back_to_run_status_file(client):
    saved = dict(app_module._new_progress(), status="completed", groups=[{"name": "n1", "status": "failed"}])
    with open(app_module.RUN_STATUS_FILE, "w") as f:
        json.dump(saved, f)
    assert client.get("/api/progress").get_json()["groups"][0]["status"] == "failed"


def test_run_start_clears_previous_run_files(client, monkeypatch):
    with open(app_module.CURRENT_RESPONSES_FILE, "w") as f:
        f.write("{}")
    monkeypatch.setitem(app_module.PIPELINE, "resize", _ok_stage)
    monkeypatch.setitem(app_module.PIPELINE, "ocr", _ok_stage)
    seen = {}
    def transcribe(group_name, on_update=None, input_dir=None, run_id=None):
        seen["current_exists_at_start"] = os.path.exists(app_module.CURRENT_RESPONSES_FILE)
        return _transcribe_factory()(group_name, on_update=on_update)
    monkeypatch.setitem(app_module.PIPELINE, "transcribe", transcribe)
    monkeypatch.setitem(app_module.PIPELINE, "export", _export)
    _post(client, [{"images": ["a.jpg"]}])
    assert seen["current_exists_at_start"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_app.py -v`
Expected: FAIL with `AttributeError: module 'app' has no attribute 'PIPELINE'`

- [ ] **Step 3: Replace the imports and globals in `app.py`**

Replace lines 1–12 with:

```python
import os
import json
import shutil
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
from dotenv import load_dotenv, set_key, dotenv_values
from PIL import Image
from pillow_heif import register_heif_opener
import fitz  # PyMuPDF

import process_images
import googlevision_translater
import note_translater
import export_responses
from pipeline_utils import atomic_write_json, INPUT_FOLDER
```

Replace lines 29–44 (the `processing_progress` block through the three folder constants) with:

```python
def _new_progress():
    return {
        'status': 'idle',
        'run_id': None,
        'total_groups': 0,
        'current_group': 0,
        'completed_groups': 0,
        'percentage': 0,
        'current_step': '',
        'groups': [],
    }


processing_progress = _new_progress()

PIPELINE = {
    'resize': process_images.process_group,
    'ocr': googlevision_translater.process_group,
    'transcribe': note_translater.process_group,
    'export': export_responses.export_run,
}

ENV_FILE = os.path.join(os.path.dirname(__file__), '.env')

OUTPUT_FOLDER = os.path.expanduser(os.getenv('OUTPUT_FOLDER', '~/Desktop/markdown_output'))
TEMP_FOLDER = os.path.expanduser(os.getenv('TEMP_FOLDER', '/tmp/transcriber/temp_uploads'))
RUN_STATUS_FILE = os.getenv('RUN_STATUS_FILE', 'run_status.json')
CURRENT_RESPONSES_FILE = os.getenv('CURRENT_RESPONSES_FILE', 'responses_current.json')
```

(`INPUT_FOLDER` now comes from `pipeline_utils` so every stage and the app agree on one directory.)

- [ ] **Step 4: Replace the `/api/process` route (lines 461–645) with the in-process loop**

```python
def _persist_progress():
    atomic_write_json(RUN_STATUS_FILE, processing_progress)


def _update_group(group_name, **fields):
    for entry in processing_progress['groups']:
        if entry['name'] == group_name:
            entry.update(fields)
            break
    _persist_progress()


def _fail_group(group_name, error, failed_groups):
    logger.warning(f'Group {group_name} failed: {error}')
    _update_group(group_name, status='failed', error=error)
    failed_groups.append(group_name)


@app.route('/api/process', methods=['POST'])
def process_images_route():
    """Run the pipeline for every group in-process; a failed group never stops the run."""
    data = request.json or {}
    groups = data.get('groups', [])
    logger.info(f"Received processing request with {len(groups)} groups")
    if not groups:
        return jsonify({'error': 'No image groups provided'}), 400

    run_id = datetime.now().isoformat(timespec='seconds')

    # Start of run: clear last run's retry window and the input folder
    if os.path.exists(CURRENT_RESPONSES_FILE):
        os.remove(CURRENT_RESPONSES_FILE)
    if os.path.exists(INPUT_FOLDER):
        shutil.rmtree(INPUT_FOLDER)
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    processing_progress.clear()
    processing_progress.update(_new_progress())
    processing_progress.update({
        'status': 'processing', 'run_id': run_id, 'total_groups': len(groups),
        'current_step': 'Starting processing',
        'groups': [{
            'name': f"n{i+1}",
            'label': f"{len(g.get('images', []))} page{'s' if len(g.get('images', [])) != 1 else ''}",
            'status': 'pending', 'stage': None,
            'pages_done': 0, 'pages_total': len(g.get('images', [])),
            'attempts': 0, 'warnings': [], 'error': None,
        } for i, g in enumerate(groups)],
    })
    _persist_progress()

    for i, group in enumerate(groups):
        group_name = f"n{i+1}"
        group_folder = os.path.join(INPUT_FOLDER, group_name)
        os.makedirs(group_folder, exist_ok=True)
        images = group.get('images', [])
        for image_file in images:
            temp_path = os.path.join(TEMP_FOLDER, image_file)
            if os.path.exists(temp_path):
                shutil.copy2(temp_path, os.path.join(group_folder, image_file))
            else:
                logger.error(f"File not found in temp: {temp_path}")
        with open(os.path.join(group_folder, 'order.json'), 'w') as f:
            json.dump({'files': images}, f, indent=2)

    results, failed_groups = [], []
    completed = 0

    for i, group in enumerate(groups):
        group_name = f"n{i+1}"
        processing_progress.update({
            'current_group': i + 1,
            'current_step': f'Processing {group_name}',
            'percentage': int((i / len(groups)) * 100),
        })
        _update_group(group_name, status='running', stage='resizing')
        try:
            stage = PIPELINE['resize'](group_name, input_dir=INPUT_FOLDER)
            if not stage.ok:
                _fail_group(group_name, f'image processing: {stage.error}', failed_groups)
                continue

            _update_group(group_name, stage='ocr')
            stage = PIPELINE['ocr'](group_name, input_dir=INPUT_FOLDER)
            if not stage.ok:
                _fail_group(group_name, f'OCR: {stage.error}', failed_groups)
                continue

            result = PIPELINE['transcribe'](
                group_name, on_update=_update_group, input_dir=INPUT_FOLDER, run_id=run_id)
            results.append(result.to_dict())
            if result.status == 'failed':
                failed_groups.append(group_name)
            else:
                completed += 1
        except Exception as e:
            logger.exception(f'Unexpected error processing {group_name}')
            _fail_group(group_name, f'unexpected error: {e}', failed_groups)
        finally:
            processing_progress.update({
                'completed_groups': completed,
                'percentage': int(((i + 1) / len(groups)) * 100),
            })
            _persist_progress()

    if completed:
        processing_progress['current_step'] = 'Exporting results'
        _persist_progress()
        exportable = [r for r in results if r['status'] in ('done', 'warning')]
        for report in PIPELINE['export'](exportable, OUTPUT_FOLDER):
            if not report['ok']:
                completed -= 1
                _fail_group(report['group_name'], f"export: {report['error']}", failed_groups)

    processing_progress['current_step'] = 'Cleaning up temporary files'
    for filename in os.listdir(TEMP_FOLDER):
        file_path = os.path.join(TEMP_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    message = f'Processing completed: {completed}/{len(groups)} groups successful'
    if failed_groups:
        message += f', {len(failed_groups)} failed'
    processing_progress.update({
        'status': 'completed', 'current_step': 'Processing complete',
        'percentage': 100, 'completed_groups': completed,
    })
    _persist_progress()

    response_data = {
        'message': message,
        'groups_processed': completed,
        'total_groups': len(groups),
        'failed_groups': failed_groups,
    }
    if completed == 0:
        return jsonify(response_data), 500
    if failed_groups:
        return jsonify(response_data), 207
    return jsonify(response_data), 200
```

Delete `import subprocess` usage entirely (it no longer appears). The old function was named `process_images` — it is renamed `process_images_route` so it does not shadow the imported module.

- [ ] **Step 5: Replace `get_progress` (lines 675–678)**

```python
@app.route('/api/progress')
def get_progress():
    """Live progress, or the last run's saved status when idle."""
    if processing_progress.get('status') == 'idle' and os.path.exists(RUN_STATUS_FILE):
        try:
            with open(RUN_STATUS_FILE) as f:
                return jsonify(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
    return jsonify(processing_progress)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_app.py -v`
Expected: 6 passed

- [ ] **Step 7: Run the full suite and a smoke import**

Run: `$PY -m pytest -q && $PY -c "import app; print('app imports ok')"`
Expected: all passed, `app imports ok`

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "run pipeline in-process with per-group status and run_status.json

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: UI — group status list, confirm-before-run, last run on load

**Files:**
- Modify: `templates/index.html` — CSS block near line 227 (`.progress-container`), markup after the `progress-steps` div (ends line 414), `DOMContentLoaded` handler at line 474, `processGroups()` at line 982, `updateProgressFromServer()` at line 1602, `processWithProgress()` at line 1641.

**Interfaces:**
- Consumes: `/api/progress` payload from Task 11.
- Produces: `renderGroupStatus(groups)`, `lastProgress` global; the Run button confirms when the last run has failed groups.

- [ ] **Step 1: Add CSS**

After the `.progress-step.completed .progress-step-label` rule (line 241), add:

```css
        .group-status-list { margin-top: 15px; display: flex; flex-direction: column; gap: 6px; }
        .group-status-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 6px; background: #f7f7f7; font-size: 14px; transition: opacity 0.3s; }
        .group-status-row .gs-name { font-weight: bold; min-width: 40px; }
        .group-status-row .gs-label { color: #666; min-width: 70px; }
        .group-status-row .gs-detail { flex: 1; color: #444; }
        .group-status-row .gs-attempts { color: #888; font-size: 12px; }
        .group-status-row.pending { color: #999; }
        .group-status-row.running { background: #e8f1ff; color: #0b5ed7; }
        .group-status-row.done { opacity: 0.5; }
        .group-status-row.warning { background: #fff4e0; color: #8a5a00; }
        .group-status-row.failed { background: #fde8e8; color: #a11a1a; }
```

- [ ] **Step 2: Add markup**

Immediately after the closing `</div>` of `progress-steps` (line 414, before the `</div>` that closes `progressContainer`), add:

```html
            <div class="group-status-list" id="groupStatusList"></div>
```

- [ ] **Step 3: Add the renderer and `lastProgress` global**

Directly above `function updateProgressFromServer(progress)` (line 1602), add:

```javascript
        let lastProgress = null;

        function renderGroupStatus(groups) {
            const list = document.getElementById('groupStatusList');
            if (!groups || groups.length === 0) { list.innerHTML = ''; return; }
            list.innerHTML = groups.map(g => {
                let detail = '';
                if (g.status === 'running') {
                    detail = `${g.stage || 'starting'} — ${g.pages_done}/${g.pages_total} pages`;
                } else if (g.status === 'failed') {
                    detail = g.error || 'failed';
                } else if (g.status === 'warning') {
                    detail = `${g.warnings.length} warning${g.warnings.length === 1 ? '' : 's'}: ${g.warnings[0] || ''}`;
                } else if (g.status === 'done') {
                    detail = 'done';
                } else {
                    detail = 'waiting';
                }
                const attempts = g.attempts ? `${g.attempts} call${g.attempts === 1 ? '' : 's'}` : '';
                return `<div class="group-status-row ${g.status}">
                    <span class="gs-name">${g.name}</span>
                    <span class="gs-label">${g.label || ''}</span>
                    <span class="gs-detail">${escapeHtml(detail)}</span>
                    <span class="gs-attempts">${attempts}</span>
                </div>`;
            }).join('');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text == null ? '' : String(text);
            return div.innerHTML;
        }
```

- [ ] **Step 4: Wire the renderer into `updateProgressFromServer`**

At the top of `updateProgressFromServer(progress)` (first line of the body), add:

```javascript
            lastProgress = progress;
            renderGroupStatus(progress.groups);
```

- [ ] **Step 5: Confirm before a run that would discard a retry**

In `processGroups()` (line 982), immediately before `processWithProgress();`, add:

```javascript
            const failedLast = (lastProgress && lastProgress.groups || []).filter(g => g.status === 'failed').length;
            if (failedLast > 0) {
                const ok = confirm(`Last run has ${failedLast} failed group${failedLast === 1 ? '' : 's'}. Running again discards the chance to retry. Continue?`);
                if (!ok) return;
            }
```

- [ ] **Step 6: Keep the status list visible after a run with problems, and fetch the final state**

In `processWithProgress()`, replace the block from `// Final update` through the `setTimeout(() => { hideProgressBar(); }, 2000);` with:

```javascript
                // Final update from the server so the list shows the last state
                try {
                    const finalProgress = await (await fetch('/api/progress')).json();
                    updateProgressFromServer(finalProgress);
                } catch (e) {
                    console.error('Failed to fetch final progress:', e);
                }
                updateProgress(5, 100, data.message || 'Processing complete!');

                if (data.message) {
                    const level = (data.failed_groups && data.failed_groups.length) ? 'error' : 'success';
                    showStatus(data.message, level, true);
                    uploadedImages = [];
                    imageGroups = [];
                    displayImages();
                    displayGroups();

                    const problems = (lastProgress && lastProgress.groups || [])
                        .some(g => g.status === 'failed' || g.status === 'warning');
                    if (!problems) {
                        setTimeout(() => { hideProgressBar(); }, 2000);
                    }
                }
```

Also change the `if (!response.ok) { throw ... }` check so a 207 (partial) and a 500 (all failed) still render the per-group list instead of throwing: replace

```javascript
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
```

with

```javascript
                if (!response.ok && response.status !== 207 && response.status !== 500) {
                    throw new Error('Network response was not ok');
                }
```

- [ ] **Step 7: Show the last run on page load**

In the `DOMContentLoaded` handler (line 474), after `startTempFolderPolling();`, add:

```javascript
            fetch('/api/progress')
                .then(r => r.json())
                .then(progress => {
                    if (progress.status === 'completed' && progress.groups && progress.groups.length) {
                        lastProgress = progress;
                        showProgressBar();
                        updateProgressFromServer(progress);
                        document.getElementById('progressHeader').textContent = `Last run (${progress.run_id})`;
                    }
                })
                .catch(e => console.error('Failed to load last run:', e));
```

- [ ] **Step 8: Verify in the browser**

Run the server from the project directory:

`~/.pyenv/versions/3.11.8/envs/media_handler/bin/flask run --host=0.0.0.0 --port=5001`

Then, using the `/browse` skill against `http://localhost:5001`:
1. Upload one small photo of handwriting, create one group, click **Process All Groups**. Watch the list show `n1` as `running` with a stage, then `done` (faded) or `warning` (amber).
2. Reload the page — the progress container appears with header `Last run (…)` and the same row.
3. Confirm `run_status.json` and `metrics.jsonl` exist in the project root and `metrics.jsonl` has one line per API call made.
4. Simulate a failure without touching credentials: upload a 0-byte `.jpg` as its own group alongside a real one. The bad group shows `failed` in red with `image processing: …`, the real group still completes, and the response is a 207.
5. With a failed group showing, click **Process All Groups** again with a new group: the confirm dialog appears.

If any of the above does not behave, fix before committing. Report which checks were observed.

- [ ] **Step 9: Commit**

```bash
git add templates/index.html
git commit -m "show per-group status list, confirm before discarding a retry, restore last run on load

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Remove the old scripts and finish housekeeping

**Files:**
- Delete: `gpt4-note-translater.py`, `googlevision-translater.py`
- Modify: `transpose_notes.sh`, `README.md`, `TODOS.md`, `docs/superpowers/specs/2026-09-15-resilient-pipeline-design.md`

- [ ] **Step 1: Confirm nothing references the old filenames**

Run: `grep -rn "gpt4-note-translater\|googlevision-translater\|subprocess" --include="*.py" --include="*.sh" --include="*.html" . | grep -v "^./docs/"`
Expected: only lines inside the two files being deleted (or none).

- [ ] **Step 2: Delete the old scripts**

```bash
git rm gpt4-note-translater.py googlevision-translater.py
```

- [ ] **Step 3: Update `transpose_notes.sh`**

Replace the four `python3 …` lines with:

```bash
python3 process_images.py
python3 googlevision_translater.py
python3 note_translater.py
python3 export_responses.py
```

- [ ] **Step 4: Update `TODOS.md`**

Mark these as done by prefixing the heading with `DONE - `: `[TODO-1]`, `[TODO-2]`, `[TODO-3]`, `[TODO-5]`. Under `[TODO-5]` append one line: `**Status:** pytest suite in tests/ — run with the media_handler interpreter: `~/.pyenv/versions/3.11.8/envs/media_handler/bin/python -m pytest`.`

- [ ] **Step 5: Update `README.md`**

Replace the `## Usage` section with:

```markdown
## Usage

Expecting traffic on 5001.

```bash
flask run --host=0.0.0.0 --port=5001
```

### Tests

```bash
~/.pyenv/versions/3.11.8/envs/media_handler/bin/python -m pytest
```

### Files written by a run

- `responses_current.json`, `run_status.json` — last run only; overwritten at the start of the next run. A failed group's results stay here so it can be retried.
- `metrics.jsonl` — one line per API call (model, prompt hash, attempts, tokens, latency). Never truncated.
- `responses.json` — append-only history of every group result.
```

- [ ] **Step 6: Align the spec with one implementation choice**

In `docs/superpowers/specs/2026-09-15-resilient-pipeline-design.md`:
- Change the `PageResult.meta` comment from `# None when status == "failed" before any response` to `# always present; a failed call records the usage it consumed`. (Implementation always writes a metrics row, which is what the Metrics section already requires.)
- In the Section 1 module table, change `export_run(results) -> list[Path]` to `export_run(results, output_dir) -> list[dict]` (one `{group_name, ok, path, error}` per attempted group), which is what lets `app.py` mark an export failure on a single group.

- [ ] **Step 7: Run the full suite one last time and the CLI smoke path**

Run: `$PY -m pytest -q && $PY -c "import process_images, googlevision_translater, note_translater, export_responses, app; print('all modules import')"`
Expected: all passed, `all modules import`

- [ ] **Step 8: Commit**

```bash
git add -A transpose_notes.sh README.md TODOS.md docs/superpowers/specs/2026-09-15-resilient-pipeline-design.md
git commit -m "remove superseded pipeline scripts; document tests and run files

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

(`git rm` in Step 2 already staged the deletions.)
