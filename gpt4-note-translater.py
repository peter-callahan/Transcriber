import base64
import json
import os
import sys
import openai
import hashlib
import unicodedata
import re
from pathlib import Path
from datetime import datetime

from PIL import Image
from obsidian_tags import load_saved_tags
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def add_suffix_to_path(file_path, suffix):
    directory, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    new_file_name = f"{base_name}{suffix}{ext}"
    return os.path.join(directory, new_file_name)


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

# Create the prompt
base_prompt = '''I am sending you one or more handwritten notes please transcribe them. If there are multiple notes
treat them as if they were one continuous note and do not summarize them or speculate about their connection.
Each individual image is accompanied by extracted text from Google Vision API. They have the same filename.
Use this text file to assist you in transcribing the words accurately. If you see a word crossed out in the image, ignore it.
I want you to add an additional annotation that includes the date of the text (if present)
and provide between 1 and 3 tags that indicates the approximate subject matter. Also, create a title for the note that helps
something about the note that will make it more memorable to me.
Output everything in markdown format, but don't include ```markdown code tags, just use the markdown syntax.
(DO NOT wrap in markdown code blocks, return ONLY the raw JSON).'''

tag_guidance = ""
if obsidian_tags:
    tag_guidance = f'''
When choosing tags, consider using tags from this existing vocabulary when appropriate:
{", ".join(obsidian_tags)}'''

prompt = base_prompt + tag_guidance + '''
The output should be in valid JSON format and in the following structure:
{
  "title": "Your title here",
  "transcription": "Your transcription in markdown format here",
  "date": "Date here (if present, use earliest date if more than one are present)",
  "tags": ["Tag1", "Tag2"]
}'''


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
    print(f"Processing GPT conversion for group: {group_name}")
else:
    # Process all folders (original behavior)
    folders_to_process = [d for d in os.listdir(
        input_images_dir) if os.path.isdir(os.path.join(input_images_dir, d))]
    print("Processing GPT conversion for all groups")

for folder in folders_to_process:
    folder_path = os.path.join(input_images_dir, folder)

    # Skip if folder doesn't exist (for single group mode)
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Group folder {folder} not found, skipping")
        continue

    folder_texts = []
    folder_images = []
    image_list = []

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
            image_list.append(image_file)
            base64_image = encode_image(image_path)
            folder_images.append({"type": "image_url", "image_url": {
                                 "url": f"data:image/jpeg;base64,{base64_image}"}})

            # Create the text file path
            extracted_text_file = create_text_path(image_path)
            if os.path.exists(extracted_text_file):
                with open(extracted_text_file, "r") as file:
                    extracted_text = file.read()
                    folder_texts.append(extracted_text)

    if folder_texts or folder_images:
        combined_content = []

        # Add the prompt as the first piece of content
        combined_content.append({"type": "text", "text": prompt})

        # Pair each image with its corresponding text
        for image, text in zip(folder_images, folder_texts):
            combined_content.append({"type": "text", "text": text})
            combined_content.append(image)

        # Add any remaining images or texts if they are unmatched
        for remaining_image in folder_images[len(folder_texts):]:
            combined_content.append(remaining_image)

        for remaining_text in folder_texts[len(folder_images):]:
            combined_content.append(
                {"type": "text", "text": remaining_text})

        uuid = generate_uuid(image_list, model)

        # Check if UUID already exists in cache
        if uuid in responses:
            print(f"UUID {uuid} found in cache. Serving from cache.")
            response_dict = responses[uuid]
        else:
            if mock_mode:
                response_dict = get_mock_response()
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": combined_content
                        }
                    ],
                    max_tokens=10000
                )

                response_dict = response.model_dump()

            print(response_dict)

            metadata_dict = response_dict['choices'][0]['message']['content']

            # Clean the response content comprehensively
            content = clean_json_text(metadata_dict)

            try:
                metadata_dict = json.loads(content)
                is_valid_json = True
            except json.JSONDecodeError as e:
                is_valid_json = False
                print(
                    f"WARNING!!! The response content is not valid JSON. {e}")
                print(f"Raw content: {repr(metadata_dict)}")
                print(f"Cleaned content: {repr(content)}")
                # Print the specific error location if available
                if hasattr(e, 'lineno') and hasattr(e, 'colno'):
                    lines = content.split('\n')
                    if e.lineno <= len(lines):
                        problem_line = lines[e.lineno - 1]
                        print(
                            f"Error at line {e.lineno}, column {e.colno}: {problem_line}")
                        print(
                            f"Error character: {repr(problem_line[e.colno-1:e.colno+10] if e.colno <= len(problem_line) else 'EOF')}")
                metadata_dict = {}

            responses[uuid] = response_dict
            responses[uuid]['is_valid_json'] = is_valid_json
            responses[uuid]['metadata'] = {
                'title': metadata_dict.get('title'),
                'transcription': metadata_dict.get('transcription'),
                'date': metadata_dict.get('date'),
                'tags': metadata_dict.get('tags'),
                'upload_date': datetime.now().isoformat(),
            }

            # Add image paths to the response dictionary
            response_dict["image_paths"] = [os.path.join(
                folder_path, img) for img in image_list]

            with open(responses_file, "w") as json_file:
                json.dump(responses, json_file, indent=4)

            print(
                f"Response for folder {folder} saved and appended to JSON file.")
