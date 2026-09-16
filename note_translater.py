import os
import re
import json
import base64
import hashlib
import time
import unicodedata
from datetime import datetime

from obsidian_tags import load_saved_tags
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

DATE_FORMAT = "%Y_%m_%d"
DATE_FORMAT_DISPLAY = "YYYY_MM_DD"
MIN_TRANSCRIPTION_LENGTH = 20
INLINE_MARKER_RE = re.compile(r"\[\?[^\]]*\]")


# ---- verbatim from gpt4-note-translater.py -------------------------------
# parse_date_string, generate_uuid, create_text_path, clean_json_text, encode_image
# (paste here, unchanged)

def parse_date_string(date_str):
    """Try multiple formats and coerce to configured DATE_FORMAT (YYYY_MM_DD).

    Handles full dates and partial dates (month/year only).
    Partial dates are coerced to the first day of the month (YYYY_MM_01).
    """
    # Full date formats (with day)
    full_date_formats = [
        "%Y_%m_%d",      # 2025_08_01 (our target format)
        "%Y-%m-%d",      # 2025-08-01
        "%d-%b-%Y",      # 1-Aug-2025
        "%d/%m/%Y",      # 01/08/2025
        "%m/%d/%Y",      # 08/01/2025
        "%d %b %Y",      # 1 Aug 2025
        "%b %d, %Y",     # Aug 1, 2025
        "%Y.%m.%d",      # 2025.08.01
        "%d.%m.%Y",      # 01.08.2025
    ]

    # Partial date formats (month/year only - will default to day 1)
    partial_date_formats = [
        "%b %Y",         # Aug 2025
        "%B %Y",         # August 2025
        "%b-%Y",         # Aug-2025
        "%B-%Y",         # August-2025
        "%m/%Y",         # 08/2025
        "%m-%Y",         # 08-2025
        "%Y-%m",         # 2025-08
        "%Y/%m",         # 2025/08
    ]

    # Try full date formats first
    for fmt in full_date_formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime(DATE_FORMAT)
        except Exception:
            continue

    # Try partial date formats (month/year only)
    for fmt in partial_date_formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            # Force day to 01 for partial dates
            dt = dt.replace(day=1)
            return dt.strftime(DATE_FORMAT)
        except Exception:
            continue

    return None  # Could not parse


def generate_uuid(filenames, model):
    if not filenames:
        raise ValueError("Filenames list is empty. Cannot generate UUID.")

    # Combine all filenames and the model to create a unique identifier
    unique_string = f"{'-'.join(sorted(set(filenames)))}-{model}"
    # print(f"Generated unique string for UUID: {unique_string}")
    return hashlib.md5(unique_string.encode()).hexdigest()


def create_text_path(image_path):
    # Replace the image file suffix with .txt
    base_name, _ = os.path.splitext(image_path)
    return f"{base_name}.txt"


def clean_json_text(text):
    """Clean text to make it valid JSON by handling various problematic characters."""
    # Remove markdown code blocks
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]  # Remove ```json
    elif text.startswith('```'):
        text = text[3:]   # Remove ```

    if text.endswith('```'):
        text = text[:-3]  # Remove closing ```

    text = text.strip()

    # Method 1: Unicode normalization (converts composed characters to decomposed)
    # This handles many Unicode issues including smart quotes
    text = unicodedata.normalize('NFKD', text)

    # Method 2: Replace common problematic characters
    replacements = {
        # Smart quotes
        '"': '"', '"': '"', ''': "'", ''': "'",
        # Em and en dashes
        '—': '-', '–': '-',
        # Other common problematic characters
        '…': '...',  # ellipsis
        '‚': ',',    # single low-9 quotation mark
        '„': '"',    # double low-9 quotation mark
        '‹': '<', '›': '>',  # single guillemets
        '«': '"', '»': '"',  # double guillemets
        # Non-breaking spaces and other whitespace
        '\xa0': ' ',  # non-breaking space
        '\u2028': '\n',  # line separator
        '\u2029': '\n\n',  # paragraph separator
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Method 3: Remove or replace any remaining problematic control characters
    # Keep only printable ASCII + newlines, tabs, and common Unicode
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Method 4: Escape literal newlines/tabs inside JSON string values.
    # The model correctly preserves line breaks from handwriting, but raw \n/\r/\t
    # inside a JSON string value are invalid — they must be escaped as \\n etc.
    # Walk char-by-char tracking string context so we only touch chars inside "...".
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and in_string:
            # Already-escaped sequence — copy both chars and skip ahead
            result.append(ch)
            if i + 1 < len(text):
                i += 1
                result.append(text[i])
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            result.append('\\r')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
        i += 1
    text = ''.join(result)

    return text


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
