# TODOS

## P1 — High Priority

### [TODO-1] Add retry with exponential backoff to call_api()
**What:** Wrap call_api() in gpt4-note-translater.py with retry logic using tenacity.
**Why:** A single rate-limit or timeout silently fails an entire page or crashes a multi-page group. One transient API error = lost transcription work.
**Context:** call_api() at line 290 has no error handling. The TODO comment at line 595 also notes invalid JSON should be retried. Use `pip install tenacity` and a `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))` decorator.
**Effort:** S

---

### [TODO-3] Refactor subprocess orchestration to in-process function calls
**What:** Each pipeline script (process_images.py, googlevision-translater.py, gpt4-note-translater.py) exposes a `process_group(group_name)` function. app.py calls them directly instead of via subprocess.run().
**Why:** subprocess.run() means no shared logging, errors are strings not exceptions, 'python' binary is hardcoded (breaks in venvs), and the pipeline can't be tested.
**Context:** Keep `__main__` blocks for CLI use. This is the prerequisite for integration tests (TODO-5). The in-process functions should return a result object, not just log.
**Effort:** M

---

### [TODO-4] Protect /api/config from exposing API keys
**What:** Strip sensitive keys (matching `*_KEY` and `*_CREDENTIALS`) from the GET /api/config response.
**Why:** The endpoint currently returns the full .env including OPENAI_API_KEY and ANTHROPIC_API_KEY. Flask runs on 0.0.0.0:5001 — anyone on the same network can read your keys.
**Context:** app.py:556. The fix is ~5 lines: filter the dotenv_values dict before returning it. The POST endpoint should only accept non-sensitive keys.
**Effort:** S

---

### [TODO-5] Add a basic test suite with pytest
**What:** Unit tests for pure functions: get_file_order, clean_json_text, parse_date_string, sanitize_filename, validate_group_output. Integration tests after TODO-3.
**Why:** Zero tests exist. Three bugs that a test would have caught have already been fixed (PNG→JPG remapping, empty individual_responses crash, empty OCR text block).
**Context:** Start with `pytest` + `pytest-mock`. No mocking needed for unit tests. Add `tests/` directory with `test_pipeline.py` and `test_export.py`.
**Effort:** M
**Depends on:** TODO-3 for integration tests (unit tests can be done independently)

---

## P2 — Medium Priority

### [TODO-2] Consolidate get_file_order() into a shared pipeline_utils.py
**What:** The function is copy-pasted identically in gpt4-note-translater.py:32, googlevision-translater.py:19, and process_images.py:22.
**Why:** A bug fix requires updating 3 files. This already caused a divergence — the PNG→JPG remapping fix was applied in 2 of 3 files at different times.
**Context:** Create `pipeline_utils.py` with `get_file_order()` and shared logging setup. All 3 scripts import from it.
**Effort:** S

---

## Vision Items (Delight Opportunities)

### [DELIGHT-1] Transcription confidence indicators in the UI
Per-page quality dots (green/yellow/red) based on validation_warnings already in responses.json. Yellow = short transcription. Red = invalid JSON. Data exists, purely a UI addition.
**Effort:** S (~30 min)

### [DELIGHT-2] Side-by-side preview: original image + transcription after processing
After processing, show a panel with the original image thumbnail on the left and transcription text on the right. Add a "Re-process" button for pages with issues. All data already available.
**Effort:** M (~75 min)

### [DELIGHT-3] Processing completion push notification to iPhone
On processing completion, POST to ntfy.sh (or similar webhook) with result summary. iPhone gets a push notification. ~15 lines. NTFY_URL configurable in .env.
**Effort:** S (~20 min)

### [DELIGHT-4] Processing history tab in the UI
A "History" tab listing all previously processed notes (title, date, tags, warnings) from responses.json. Click to view transcription inline. New Flask endpoint + read-only UI panel.
**Effort:** M (~45 min)

### [DELIGHT-5] Startup health check endpoint
`/api/health` endpoint checks that env vars are set, credentials file exists, output folder is writable. UI shows a warning banner if anything is misconfigured. Prevents silent failures.
**Effort:** S (~30 min)
