# Resilient Pipeline — Design

**Date:** 2026-09-15
**Status:** Approved in brainstorm, pending implementation plan
**Scope label:** Sub-project A of three (A: resilient pipeline, B: review UI, C: accuracy spike)

## Problem

A single API failure during transcription is invisible or fatal:

- Per-page errors in `gpt4-note-translater.py` are `except Exception: continue` — the page is silently dropped and the group exports with one page missing.
- Invalid JSON from the model is exported with the raw model text as the transcription, unflagged.
- The multi-page summary call has no error handling; an exception there kills the group's subprocess.
- `app.py` runs each stage via `subprocess.run()`, so failures arrive as stderr strings. It cannot tell "page 2 of group 3 failed" from "everything failed."
- The UI shows one progress bar and a final message. There is no per-group outcome.

Goal: transient failures are retried within a cost cap, a failed group never stops the run, and the user sees per-group status live and after the run.

## Non-goals

- Side-by-side review UI, "retry with my note" button — **sub-project B**.
- OCR replacement, dual-model disagreement, golden-set accuracy measurement — **sub-project C** (spike).
- Result caching keyed on image hash — considered, **dropped**.
- Multi-run history / `runs/<id>/` folders — considered, **dropped**. Last run only.
- Auto-retry on quality warnings (short transcription, missing date in body) — not retried; surfaced as `warning`.

## Decisions (recorded)

| Decision | Choice |
|---|---|
| Multi-page group with one page failed after retries | Whole group is `failed`, nothing exported; successful pages' results are **kept** in the result object so a later retry re-sends only the failed page. |
| Retry triggers | Transport errors, unparseable JSON, missing required fields. Not quality warnings. |
| Retry cap | 3 total API calls per page (and per summary call). |
| Retry shape | Transport error → identical redo with backoff. Parse/structure error → repair turn continuing the same conversation. |
| Uncertainty signal | Separate `uncertain` array in the model's JSON. **Never** inline markers in the transcription text; exported markdown stays clean. |
| Streaming | Extend the existing `/api/progress` poll with a `groups` array. No SSE, no background job. |
| Retention | Last run only. `run_status.json` and `responses_current.json` persist until the next run starts. Run button warns if the last run has failed groups. |
| Metrics | Every API call writes a `CallMeta` row (model, provider, prompt hash, attempts, latency, tokens, word/uncertain counts) to permanent `metrics.jsonl`. No quality score — accuracy comes from B's corrections. |

## Section 1 — Orchestration (in-process)

### Module layout

| Today | After | Entry point |
|---|---|---|
| `process_images.py` | `process_images.py` | `process_group(group_name) -> StageResult` |
| `googlevision-translater.py` | `googlevision_translater.py` | `process_group(group_name) -> StageResult` |
| `gpt4-note-translater.py` | `note_translater.py` | `process_group(group_name, on_update) -> GroupResult` |
| (copy-pasted ×3) | `pipeline_utils.py` | `get_file_order()`, logging setup, provider client, `call_api()`, `call_with_retry()` |
| `export_responses.py` | `export_responses.py` | `export_run(results, output_dir) -> list[dict]` |

Hyphenated filenames are renamed because they cannot be imported. Each module keeps an `if __name__ == "__main__":` block that parses argv and calls its `process_group`, so `transpose_notes.sh` and manual CLI use keep working.

`app.py` imports the four entry points and calls them directly inside the existing per-group loop. `subprocess.run` is removed from the processing path. Exceptions propagate as exceptions; the loop catches per group.

### Result objects

```python
@dataclass
class StageResult:
    ok: bool
    error: str | None = None

@dataclass
class PageResult:
    filename: str
    status: Literal["done", "warning", "failed"]
    attempts: int                       # API calls made for this page
    data: dict | None                   # {"title","date","transcription","tags"} when parsed
    uncertain: list[dict]               # [{"text": str, "context": str}]
    warnings: list[str]
    error: str | None                   # last error when status == "failed"
    history: list[dict]                 # per-attempt {"kind","error"} for the repair trail
    meta: CallMeta | None               # always present; a failed call records the usage it consumed

@dataclass
class CallMeta:
    run_id: str
    group_name: str
    filename: str | None                # None for the summary call
    kind: Literal["page", "summary"]
    provider: str
    model: str
    prompt_hash: str                    # sha256 of the prompt text actually sent
    attempts: int
    latency_ms: int                     # wall time across all attempts
    input_tokens: int
    output_tokens: int
    word_count: int                     # of the transcription (or summary) text
    uncertain_count: int
    timestamp: str                      # ISO 8601

@dataclass
class GroupResult:
    group_name: str
    status: Literal["done", "warning", "failed"]
    file_order: list[str]
    image_paths: list[str]
    pages: list[PageResult]
    summary: dict | None                # multi-page only; same shape rules as a page's data
    summary_attempts: int
    summary_meta: CallMeta | None
    warnings: list[str]                 # group-level validation warnings
    errors: list[str]
```

`CallMeta` is deliberately only things that are observable — there is no "quality score" field. Accuracy is not computable without ground truth; that ground truth comes from user corrections in sub-project B and is recorded against these rows.

Status roll-up: any page `failed` or summary `failed` → group `failed`. Else any warning anywhere → `warning`. Else `done`.

`GroupResult` is serialised to `responses_current.json` (keyed by the group uuid as today) and appended to `responses.json` history. `export_responses.py` reads `GroupResult` shape only — the old `individual_responses` / `normalized` shape is retired.

`on_update(group_name, **fields)` is called after each stage, each page, and each retry with the fields from Section 3.

## Section 2 — Retry & repair

`pipeline_utils.call_with_retry(content, max_tokens, parse, max_calls=3) -> Attempt`

```python
@dataclass
class Attempt:
    ok: bool
    parsed: dict | None
    attempts: int
    history: list[dict]     # [{"kind": "transport"|"parse"|"structure", "error": str}]
    error: str | None
```

`parse(text) -> dict` raises `ParseError(msg)` on bad JSON and `StructureError(msg)` on a missing/invalid required field. Each caller passes its own validator:

- **Page validator:** required `title`, `date`, `transcription` (str), `tags` (list). `uncertain` optional; if present must be a list of `{"text": str, "context": str}`. If the transcription matches `\[\?[^\]]*\]`, add a warning `"inline uncertainty marker in transcription"` — do not fail, do not strip.
- **Summary validator:** required `title`, `date`, `summary`, `tags`.

Loop (max 3 calls):

1. Call. On success return.
2. **Transport error** (provider SDK rate-limit / timeout / connection / 5xx exception types): sleep `2 ** (attempt-1)` seconds (1s, then 4s… capped at 10s), re-send identical messages.
3. **`ParseError` / `StructureError`:** append `{"role": "assistant", "content": <bad output>}` then a user message:
   > The previous response could not be used: `<error text>`. Return the same transcription in the required JSON structure exactly as specified. Do not change, correct, reorder, or omit any transcribed text.
   Re-send the full message list.
4. Any other exception (auth, bad request, unknown) → return immediately with `ok=False`, not retried.
5. After 3 calls → `ok=False`, `error` = last error, `history` full.

Messages are built provider-neutral as `[{"role", "content": [blocks]}]` and translated per provider inside `call_api()` (the existing `_to_bedrock_content` extends to multi-turn; OpenAI and Anthropic accept the list directly). `call_api(messages, max_tokens) -> (text, usage)` therefore changes signature: it takes a message list and returns the text plus a `{"input_tokens", "output_tokens"}` dict read from the provider's usage field. `call_with_retry` sums usage across attempts and measures wall time; the caller fills the rest of `CallMeta`.

`retry_with_feedback(messages, prior_output, feedback) -> Attempt` is the same loop seeded with a user-supplied `feedback` string in place of the parser error. Exposed now; first caller is sub-project B.

`mock_mode` is removed from `note_translater.py`; tests inject a fake `call_api`.

### Prompt change

The single-page prompt's JSON schema gains:

```json
"uncertain": [{"text": "<word or phrase as transcribed>", "context": "<a few surrounding words>"}]
```

with the instruction: list words you could not read with confidence; put your best reading in the transcription itself and do not mark it there. Empty list if none.

## Section 3 — Status contract & streaming

`processing_progress` (module global in `app.py`) gains `groups`, initialised before the loop:

```json
{
  "status": "processing",
  "run_id": "2026-09-15T20:31:04",
  "total_groups": 3,
  "completed_groups": 1,
  "percentage": 33,
  "current_step": "Transcribing n2",
  "groups": [
    {"name": "n1", "label": "3 pages", "status": "done",    "stage": "complete",     "pages_done": 3, "pages_total": 3, "attempts": 3, "warnings": [], "error": null},
    {"name": "n2", "label": "1 page",  "status": "running", "stage": "transcribing", "pages_done": 0, "pages_total": 1, "attempts": 2, "warnings": [], "error": null},
    {"name": "n3", "label": "2 pages", "status": "pending", "stage": null,           "pages_done": 0, "pages_total": 2, "attempts": 0, "warnings": [], "error": null}
  ]
}
```

- `status` ∈ `pending | running | done | warning | failed`
- `stage` ∈ `resizing | ocr | transcribing | summarising | complete | null`
- `attempts` = total API calls for the group so far (pages + summary)
- `error` = human-readable, e.g. `"page 2 (IMG_0412.jpg): invalid JSON after 3 attempts"`

Every mutation of `processing_progress` also writes it to `run_status.json` (atomic write: temp file + rename). `GET /api/progress` returns the in-memory dict; if in-memory `status == "idle"` and `run_status.json` exists, it returns the file instead. `POST /api/process` resets both at start.

### UI

Below the existing progress bar, a `#groupStatusList` rendered from `groups` on every poll:

- `pending` grey, `running` blue with stage text and `attempts`, `done` faded to 50% opacity, `warning` amber with warning count, `failed` red with `error` text.
- Existing bar, percentage, and step icons keep working from `completed_groups` / `current_step`.
- On page load, `fetch('/api/progress')` once: if a completed run is returned, render its list so the last outcome is visible after a refresh.
- Run button: before `POST /api/process`, if the last progress payload has any `failed` group, `confirm("Last run has N failed group(s). Running again discards the chance to retry. Continue?")`.

Polling cadence and the temp-folder poll are unchanged.

## Section 4 — Export & retention

- `export_run()` exports groups with status `done` or `warning` only. `failed` groups are skipped and logged.
- `responses_current.json` is **not** deleted after export. It and `run_status.json` persist until the next `POST /api/process`, which overwrites both at start. This is the retry window for sub-project B.
- `input_images/` is cleared at the start of the next run (today's behaviour). Failed groups' images therefore remain available for the same window.
- `temp_uploads/` is cleared at the end of every run (unchanged — pipeline plumbing only).
- History `responses.json`: every `GroupResult` is appended regardless of status; the group's `status` field is stored so failed runs are distinguishable later.

### Metrics log (permanent)

`metrics.jsonl` — append-only, one JSON line per `CallMeta`, written the moment each page or summary call finishes (success or final failure). Failed calls are logged with `status: "failed"` and whatever usage was consumed. **Nothing in the pipeline ever truncates or rewrites this file**; it is the one artefact that persists across runs regardless of the last-run wipe, so model and prompt changes can be compared over time. Same atomic-append discipline as the other writers.

Row shape is `CallMeta` plus `status` (`done | warning | failed`) and `error`. Sub-project B appends correction rows to the same file (`kind: "correction"`, keyed by `run_id + group_name + filename`, carrying the character error rate) so accuracy can be joined to model/prompt without a second store.

`metrics.jsonl` is gitignored.

## Section 5 — Error handling summary

| Failure | Where caught | Effect |
|---|---|---|
| Resize or OCR stage raises | `app.py` loop | Group `failed`, error recorded, loop continues |
| Page API transport error | `call_with_retry` | Backoff redo ×2; then page `failed` |
| Page bad JSON / missing field | `call_with_retry` | Repair turn ×2; then page `failed` |
| Page non-retryable exception | `call_with_retry` | Page `failed` immediately |
| Summary call fails | `call_with_retry` | Group `failed`; pages kept |
| Group validator warnings | `note_translater` | Group `warning`, exported |
| Export raises for one group | `export_run` | That group `failed` post-hoc, others exported |
| Unhandled exception in loop body | `app.py` | Group `failed` with traceback text, loop continues |

Nothing in the loop aborts the run. `POST /api/process` returns 200 / 207 / 500 as today based on the final counts.

## Section 6 — Testing

`tests/` with pytest. No live API calls anywhere.

- **`test_utils.py`** — `get_file_order` (missing file, missing entries, PNG→JPG remap), `clean_json_text`, `parse_date_string`, `sanitize_filename`.
- **`test_validators.py`** — page and summary validators: happy path, each missing field, `uncertain` malformed, inline marker → warning not error.
- **`test_retry.py`** — `call_with_retry` against a scripted fake `call_api`:
  - transport error → success on 2nd call, `attempts == 2`
  - bad JSON → repaired on 2nd call; asserts the 2nd message list contains the assistant turn and repair text
  - three failures → `ok == False`, `attempts == 3`, `history` has 3 entries
  - non-retryable exception → `ok == False`, `attempts == 1`
  - backoff sleep is patched
  - usage is summed across attempts; latency measured across attempts
- **`test_metrics.py`** — a page call appends exactly one well-formed line to `metrics.jsonl`; a failed page still appends with `status: "failed"`; the file is never truncated by a new run.
- **`test_note_translater.py`** — `process_group` on a fixture folder (2 images + `order.json` + OCR sidecars) with fake `call_api`:
  - all pages ok → `GroupResult.status == "done"`, summary present
  - page 2 fails all attempts → group `failed`, page 1 result retained, `on_update` called with the expected sequence
- **`test_export.py`** — `export_run` skips `failed`, exports `warning`, output folder naming.

`requirements.txt` gains `pytest`. No `tenacity` — the loop is small enough to own.

## Follow-ups (not in this spec)

- **B — Review UI:** side-by-side image/transcription, `uncertain` highlights, per-group retry via `retry_with_feedback`, only `failed` pages re-sent.
- **C — Accuracy spike:** golden set + CER script; variants: image-only (no OCR), Google OCR + model, Haiku first pass + Sonnet, Sonnet + Haiku disagreement.
- If the last-run-only retry window proves too tight in practice, revisit `runs/<id>/` retention.
