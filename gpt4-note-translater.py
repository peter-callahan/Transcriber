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


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def add_suffix_to_path(file_path, suffix):
    directory, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    new_file_name = f"{base_name}{suffix}{ext}"
    return os.path.join(directory, new_file_name)


def parse_date_string(date_str):
    """Try multiple formats and coerce to DD-MMM-YYYY."""
    date_formats = [
        "%d-%b-%Y",      # 1-Aug-2025
        "%d/%m/%Y",      # 01/08/2025
        "%Y-%m-%d",      # 2025-08-01
        "%m/%d/%Y",      # 08/01/2025
        "%d %b %Y",      # 1 Aug 2025
        "%b %d, %Y",     # Aug 1, 2025
        "%Y.%m.%d",      # 2025.08.01
        "%d.%m.%Y",      # 01.08.2025
        # Add more as needed
    ]
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%d-%b-%Y")
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
    Each response should contain parsed metadata with title, transcription, date, tags.
    """
    if not individual_responses:
        return {}

    # Collect all valid parsed responses
    valid_responses = []
    all_tags = []
    dates = []

    for response_data in individual_responses:
        valid_responses.append(response_data['transcription']['transcription'])
        all_tags.append(response_data['transcription'].get('tags', []))
        dates.append(response_data['transcription'].get('date', None))

    if not valid_responses:
        return {}

    # Combine transcriptions
    combined_transcription = "\n\n".join(
        [response for response in valid_responses])

    flat_tags = list(chain.from_iterable(all_tags))
    unique_tags = list(set(flat_tags))

    if dates:
        parsed_dates = []
        for date_str in dates:
            coerced_date = parse_date_string(date_str) if date_str else None
            if coerced_date:
                dt = datetime.strptime(coerced_date, "%d-%b-%Y")
                parsed_dates.append((dt, coerced_date))
        if parsed_dates:
            parsed_dates.sort(key=lambda x: x[0])
            earliest_date = parsed_dates[0][1]
        else:
            earliest_date = dates[0]  # Fallback to first date

    # Create combined metadata
    combined_metadata = {
        'transcription': combined_transcription,
        'date': earliest_date,
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
I want you to add an additional annotation that includes the date of the text (if present), in the <date_here> field.
'''

multi_prompt = '''Here are your directions:
I am sending you a combination of previously transcribed notes that I want you to summarize in the following ways.
Treat the incoming notes as pages in one continuous document where you should add your additional comments in the <summary_here> field.
I want you to add an additional annotation that includes the date range of the texts present and provide between 1 and 3 tags that indicates the approximate subject matter.
Create a title for the note that helps summarize the content of the included notes, make it more memorable to me if possible.
Output the text in markdown format and wrap it in JSON, using the included template below for structure. Your transcription should replace the <transcription_here> in the template.
I want you to devise a simple title, based on the content of all the notes, and insert it into the <title_here> field.
I want you to add an additional annotation that includes a date range from the notes (if present), or a single date if there is only one. This should go in the <date_here> field.
Do not include ```markdown code tags or ```json code tags, otherwise structure using normal JSON containing normal markdown syntax. (DO NOT WRAP in markdown or json code blocks).
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
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
}
'''

date_format_rules = '''
All dates should be in the format DD-MMM-YYYY, such as 1-Aug-2025.  Do not include / in dates or any characters that would disrupt their use as a filename.
'''

tag_guidance = ""
if obsidian_tags:
    tag_guidance = f'''
When choosing tags, consider using tags from this existing vocabulary when appropriate:
{", ".join(obsidian_tags)}'''

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

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
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
        response_dict = {}

        # Check if UUID already exists in cache
        if uuid in responses:
            logger.info(f"UUID {uuid} found in cache. Serving from cache.")
            response_dict = responses[uuid]

        else:
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

                    individual_responses.append({
                        'response': single_response,
                        'transcription': cleaned_content,
                        'is_valid_json': is_valid_json,
                        'filename': pair['filename']
                    })

                except Exception as e:
                    logger.error(f"Error processing {pair['filename']}: {e}")
                    continue

            # Add summary details for multiple responses
            response_dict[uuid] = {}

            if len(individual_responses) > 1:

                all_individual_responses = combine_responses(
                    individual_responses)

                combined_metadata = [
                    {"type": "text", "text": multi_response_prompt},
                    {"type": "text", "text": json.dumps(
                        all_individual_responses, indent=2)}
                ]

                # todo: make secondary API call that uses the individual calls as an input, the purpose is to add tags and metadata at the level of the combined pages

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": combined_metadata
                        }
                    ],
                    max_tokens=16384
                )
                multi_response = response.model_dump()

                logger.info(
                    f"Combined GPT Response: {len(individual_responses)} individual responses processed")

                response_dict[uuid]["image_paths"] = [os.path.join(
                    folder_path, img) for img in image_list]

                response_content = multi_response['choices'][0]['message']['content']
                cleaned_content = clean_json_text(response_content)

                response_dict[uuid]['summary'] = {}

                try:
                    parsed_content = json.loads(cleaned_content)
                    response_dict[uuid]['summary']['is_valid_json'] = True
                    response_dict[uuid]['summary']['contents'] = parsed_content

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in final response: {e}")
                    response_dict[uuid]['summary']['is_valid_json'] = False
                    response_dict[uuid]['summary']['contents'] = response_content

            response_dict[uuid]['individual_responses'] = individual_responses

            with open(responses_file, "w") as json_file:
                json.dump(response_dict, json_file, indent=4)

            logger.info(
                f"Response for folder {folder} saved and appended to JSON file.")
