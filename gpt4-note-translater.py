import base64
import json
import os
import sys
import openai
import hashlib
import unicodedata
import re
import logging
from pathlib import Path
from datetime import datetime

from PIL import Image
from obsidian_tags import load_saved_tags
from dotenv import load_dotenv
from itertools import chain

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Date configuration - centralized date format control
DATE_FORMAT = "%Y_%m_%d"  # Output format: YYYY_MM_DD
DATE_FORMAT_DISPLAY = "YYYY_MM_DD"  # Human-readable format for prompts


def get_file_order(folder_path):
    """
    Read the intended file processing order from order.json.
    Falls back to sorted filenames if order.json doesn't exist.

    Args:
        folder_path: Path to the group folder

    Returns:
        List of filenames in the order they should be processed
    """
    order_file = os.path.join(folder_path, 'order.json')

    if os.path.exists(order_file):
        try:
            with open(order_file, 'r') as f:
                order_data = json.load(f)
                return order_data.get('files', [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"Failed to read order.json: {e}, falling back to sorted order")

    # Fallback: return sorted list of image files
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.lower().endswith(
        ('.jpg', '.jpeg', '.png', '.heic'))]
    return sorted(image_files)


def validate_group_output(folder, folder_path, file_order, individual_responses, summary=None):
    """
    Run structural checks on processed output and log warnings for any problems found.
    Returns a list of warning strings (empty = all clear).
    """
    warnings = []

    # --- Order check: files on disk vs order.json ---
    # Normalize all filenames to .jpg stem for comparison, since process_images.py
    # converts PNG/HEIC to JPEG and removes the original.
    def stem_to_jpg(filename):
        return os.path.splitext(filename)[0] + '.jpg'

    all_files_on_disk = set(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))
    )
    disk_normalized = set(stem_to_jpg(f) for f in all_files_on_disk)
    order_normalized = set(stem_to_jpg(f) for f in file_order)
    missing_from_order = disk_normalized - order_normalized
    missing_from_disk = order_normalized - disk_normalized

    order_source = "order.json" if os.path.exists(os.path.join(folder_path, 'order.json')) else "sorted fallback"
    if order_source == "sorted fallback":
        warnings.append(f"No order.json found — using alphabetical sort. Pages may be out of order.")
    if missing_from_order:
        warnings.append(f"Files on disk not in order list (will be skipped): {sorted(missing_from_order)}")
    if missing_from_disk:
        warnings.append(f"Files in order list not found on disk (will be missing): {sorted(missing_from_disk)}")

    # --- JSON validity rate ---
    total = len(individual_responses)
    invalid = [r['filename'] for r in individual_responses if not r.get('is_valid_json')]
    if invalid:
        warnings.append(f"Invalid JSON from model for {len(invalid)}/{total} pages: {invalid}")

    # --- Empty/suspiciously short transcriptions ---
    MIN_TRANSCRIPTION_LENGTH = 20
    for r in individual_responses:
        text = r.get('normalized', {}).get('transcription', '')
        if len(text.strip()) < MIN_TRANSCRIPTION_LENGTH:
            warnings.append(f"Suspiciously short transcription for {r['filename']!r} ({len(text.strip())} chars) — may be a failed page")

    # --- Tag count ---
    for r in individual_responses:
        tags = r.get('normalized', {}).get('tags', [])
        if len(tags) > 3:
            warnings.append(f"Too many tags ({len(tags)}) returned for {r['filename']!r} — expected 3 max: {tags}")

    if summary:
        contents = summary.get('contents', {})
        if summary.get('is_valid_json') and isinstance(contents, dict):
            summary_tags = contents.get('tags', [])
            if len(summary_tags) > 3:
                warnings.append(f"Summary response returned {len(summary_tags)} tags — expected 3 max: {summary_tags}")
            continuous = contents.get('continuous_transcription', '')
            if len(continuous.strip()) < MIN_TRANSCRIPTION_LENGTH:
                warnings.append(f"Continuous transcription in summary is suspiciously short ({len(continuous.strip())} chars)")
        elif not summary.get('is_valid_json'):
            warnings.append("Summary (multi-page) response returned invalid JSON")

    # Log all warnings
    if warnings:
        logger.warning(f"[VALIDATION] Group {folder!r} — {len(warnings)} issue(s) found:")
        for w in warnings:
            logger.warning(f"  - {w}")
    else:
        logger.info(f"[VALIDATION] Group {folder!r} — all checks passed")

    return warnings


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def add_suffix_to_path(file_path, suffix):
    directory, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    new_file_name = f"{base_name}{suffix}{ext}"
    return os.path.join(directory, new_file_name)


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

    return text


client = openai.OpenAI()

# Add a flag to enable mock mode
mock_mode = False

# Define a mock response


def get_mock_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "# Mock Title\n\nThis is a mock transcription of the handwritten notes.\n\n- Mock annotation: Date: 2025-07-25\n- Mock tag: Example Subject"
                }
            }
        ]
    }


def combine_responses(individual_responses):
    """
    Combine multiple individual GPT responses into a single unified response.
    Uses pre-normalized data from individual responses (single source of truth).
    """
    if not individual_responses:
        return {}

    # Collect all valid parsed responses using normalized data
    valid_responses = []
    all_tags = []
    dates = []

    for response_data in individual_responses:
        # Use pre-normalized data (created when response was saved)
        normalized = response_data.get('normalized', {})

        transcription_text = normalized.get('transcription', '')
        if transcription_text:
            valid_responses.append(transcription_text)

        all_tags.append(normalized.get('tags', []))

        date_str = normalized.get('date', None)
        if date_str:
            dates.append(date_str)

    if not valid_responses:
        return {}

    # Combine transcriptions
    combined_transcription = "\n\n".join(valid_responses)

    # Combine tags
    flat_tags = list(chain.from_iterable(all_tags))
    unique_tags = list(set(flat_tags))

    # Process dates - normalize all to DATE_FORMAT and collect them
    parsed_dates = []
    if dates:
        for date_str in dates:
            if date_str:
                normalized_date = parse_date_string(date_str)
                if normalized_date:
                    parsed_dates.append(normalized_date)

    # Create combined metadata
    combined_metadata = {
        'transcription': combined_transcription,
        'date': parsed_dates if parsed_dates else [],
        'tags': unique_tags
    }

    return combined_metadata


# Load config with fallback
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    input_images_dir = os.path.expanduser(config['input_folder'])
except FileNotFoundError:
    # Default fallback when config.json doesn't exist
    input_images_dir = "input_images"
image_paths = []

for root, _, files in os.walk(input_images_dir):
    for image_file in files:
        image_path = os.path.join(root, image_file)
        if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
            image_paths.append(image_path)

responses_file = os.getenv('RESPONSES_FILE', 'responses.json')

model = "gpt-4o"

# Get existing Obsidian tags to guide AI tagging
obsidian_tags_file = os.getenv('OBSIDIAN_TAGS_FILE', 'obsidian_tags.json')
obsidian_tags = load_saved_tags(obsidian_tags_file)

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
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
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

tag_guidance = ""
if obsidian_tags:
    tag_guidance = f'''
When choosing tags, consider using tags from this existing vocabulary when appropriate:
{", ".join(obsidian_tags)}

CRITICAL - Tag Limit:
- Provide EXACTLY 3 tags maximum - choose the 3 most relevant tags only
- If fewer than 3 tags are appropriate, provide fewer
- DO NOT exceed 3 tags under any circumstances'''

single_response_prompt = single_image_prompt + \
    date_format_rules + expected_format_single_prompt + tag_guidance

multi_response_prompt = multi_prompt + date_format_rules + \
    expected_format_multiprompt + tag_guidance

try:
    with open(responses_file, "r") as json_file:
        responses = json.load(json_file)
except (FileNotFoundError, json.JSONDecodeError):
    responses = {}


# Check if specific group was provided as argument
if len(sys.argv) > 1:
    group_name = sys.argv[1]
    # Process only the specified group folder
    folders_to_process = [group_name]
    logger.info(f"Processing GPT conversion for group: {group_name}")
else:
    # Process all folders (original behavior)
    folders_to_process = [d for d in os.listdir(
        input_images_dir) if os.path.isdir(os.path.join(input_images_dir, d))]
    logger.info("Processing GPT conversion for all groups")

for folder in folders_to_process:
    folder_path = os.path.join(input_images_dir, folder)

    # Skip if folder doesn't exist (for single group mode)
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        logger.warning(f"Group folder {folder} not found, skipping")
        continue

    # Collect image and text pairs
    image_text_pairs = []
    image_list = []

    # Get files in the correct order from order.json
    file_order = get_file_order(folder_path)
    logger.info(
        f"Processing GPT for group {folder}, {len(file_order)} files in order: {file_order}")

    for image_file in file_order:
        image_path = os.path.join(folder_path, image_file)

        # process_images.py converts non-JPEG files to .jpg — remap if original is gone
        if not os.path.exists(image_path):
            jpg_path = os.path.join(folder_path, os.path.splitext(image_file)[0] + ".jpg")
            if os.path.exists(jpg_path):
                logger.info(f"Original {image_file} not found, using converted {os.path.basename(jpg_path)}")
                image_file = os.path.basename(jpg_path)
                image_path = jpg_path
            else:
                logger.warning(f"Skipping {image_file} — file not found and no .jpg equivalent exists")
                continue

        if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
            logger.info(f"Processing GPT conversion for: {image_file}")
            image_list.append(image_file)

            # Encode image
            base64_image = encode_image(image_path)
            image_data = {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"}}

            # Get corresponding text
            extracted_text = ""
            extracted_text_file = create_text_path(image_path)
            if os.path.exists(extracted_text_file):
                with open(extracted_text_file, "r") as file:
                    extracted_text = file.read()

            image_text_pairs.append({
                'image': image_data,
                'text': extracted_text,
                'filename': image_file
            })

    if image_text_pairs:
        uuid = generate_uuid(image_list, model)

        # Check if UUID already exists in cache
        if uuid in responses:
            logger.info(f"UUID {uuid} found in cache. Serving from cache.")
            continue

        # Process each image/text pair individually
        individual_responses = []

        for i, pair in enumerate(image_text_pairs):
            logger.info(
                f"Processing image {i+1}/{len(image_text_pairs)}: {pair['filename']}")

            # Create content for this single image
            single_content = [
                {"type": "text", "text": single_response_prompt},
                {"type": "text", "text": pair['text']},
                pair['image']
            ]

            try:
                if mock_mode:
                    single_response = get_mock_response()
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": single_content
                            }
                        ],
                        max_tokens=10000
                    )
                    single_response = response.model_dump()

                # Parse the individual response
                content = single_response['choices'][0]['message']['content']
                cleaned_content = clean_json_text(content)

                try:
                    cleaned_content = json.loads(cleaned_content)
                    is_valid_json = True
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Invalid JSON in response for {pair['filename']}: {e}")

                    # todo: Do something to handle this, try again, or maybe skip??

                    is_valid_json = False
                    # metadata_dict = {}

                # Create normalized data for export (single source of truth)
                if is_valid_json and isinstance(cleaned_content, dict):
                    normalized = {
                        'date': cleaned_content.get('date', ''),
                        'transcription': cleaned_content.get('transcription', ''),
                        'tags': cleaned_content.get('tags', [])
                    }
                else:
                    # Invalid JSON - use raw text
                    normalized = {
                        'date': '',
                        'transcription': cleaned_content if isinstance(cleaned_content, str) else str(cleaned_content),
                        'tags': []
                    }

                individual_responses.append({
                    'response': single_response,
                    'transcription': cleaned_content,
                    'is_valid_json': is_valid_json,
                    'filename': pair['filename'],
                    'normalized': normalized  # Pre-processed data for export
                })

            except Exception as e:
                logger.error(f"Error processing {pair['filename']}: {e}")
                continue

        # Prepare data for this UUID
        responses[uuid] = {}
        responses[uuid]["image_paths"] = [os.path.join(
            folder_path, img) for img in image_list]
        responses[uuid]["file_order"] = file_order
        responses[uuid]["group_name"] = folder
        responses[uuid]['individual_responses'] = individual_responses

        # Add summary details for multiple responses
        if len(individual_responses) > 1:

            # Build content with ALL images + OCR hints
            combined_metadata = [
                {"type": "text", "text": multi_response_prompt}
            ]

            # Add all images and OCR text in order
            for pair in image_text_pairs:
                combined_metadata.append(pair['image'])
                if pair['text']:
                    combined_metadata.append(
                        {"type": "text", "text": f"\n\nOCR text:\n{pair['text']}\n"})

            logger.info(
                f'Outgoing multi-response API call with {len(image_text_pairs)} images')

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": combined_metadata  # type: ignore
                    }
                ],
                max_tokens=16384
            )
            multi_response = response.model_dump()

            logger.info(
                f"Combined GPT Response: {len(individual_responses)} individual responses processed")

            response_content = multi_response['choices'][0]['message']['content']
            cleaned_content = clean_json_text(response_content)

            responses[uuid]['summary'] = {}

            try:
                parsed_content = json.loads(cleaned_content)
                responses[uuid]['summary']['is_valid_json'] = True
                responses[uuid]['summary']['contents'] = parsed_content

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in final response: {e}")
                responses[uuid]['summary']['is_valid_json'] = False
                responses[uuid]['summary']['contents'] = response_content

        # Validate output before saving
        warnings = validate_group_output(
            folder, folder_path, file_order, individual_responses,
            summary=responses[uuid].get('summary')
        )
        if warnings:
            responses[uuid]['validation_warnings'] = warnings

        # Save all responses (not overwrite)
        with open(responses_file, "w") as json_file:
            json.dump(responses, json_file, indent=4)

        logger.info(
            f"Response for folder {folder} saved and appended to JSON file.")
        logger.info(f"Successfully completed group {folder}")
