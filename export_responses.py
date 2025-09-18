import os
import shutil
import json
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_filename(filename):
    """
    Sanitize a string to be safe for use as a filename/folder name.
    Removes or replaces characters that are invalid in file paths.
    """
    if not filename:
        return 'Untitled'

    # Replace problematic characters with underscores
    # This includes: / \ : * ? " < > |
    sanitized = re.sub(r'[/\\:*?"<>|]', '_', filename)

    # Replace multiple underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores and spaces
    sanitized = sanitized.strip('_ ')

    # Ensure we have a valid filename
    if not sanitized:
        sanitized = 'Untitled'

    return sanitized


# Load config with fallback
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    output_dir = config.get('output_folder', './markdown_output')
except FileNotFoundError:
    # Default fallback when config.json doesn't exist
    output_dir = "./markdown_output"

# Expand tilde (~) for home directory if present
output_dir = os.path.expanduser(output_dir)

# Create the output directory
os.makedirs(output_dir, exist_ok=True)

# Load the JSON data
with open("responses.json", "r") as json_file:
    responses = json.load(json_file)

# Iterate through each response
for uuid, response_data in responses.items():

    title = ''
    date = ''
    summary = ''
    tags = ''

    individual_responses = response_data['individual_responses']

    # Properly setting up title/date/metadata depending on nature of note upload
    multi_note_upload = False

    if response_data.get('summary', {}).get('is_valid_json'):
        multi_note_upload = True
        logger.info(f"Mutlipart note detected: {uuid}")

        # Have a summary, add this to the top of the extraction
        title = response_data['summary']['contents'].get('title', 'Untitled')
        date = response_data['summary']['contents'].get('date', 'Unknown Date')
        summary = response_data['summary']['contents'].get('summary', '')
        tags = response_data['summary']['contents'].get('tags', [])

        # Create safe folder name with proper null handling
        safe_title = sanitize_filename(title) if title else 'Untitled'
        safe_date = sanitize_filename(date) if date else 'Unknown_Date'
        folder_name = f"{safe_title}_{safe_date}"

    elif response_data.get('individual_responses')[0]['is_valid_json']:
        logger.info(f"Single note detected: {uuid}")
        # Handle individual responses if present
        single_note_object = response_data.get('individual_responses')[
            0].get('transcription', '')

        title = single_note_object.get('title', '')
        date = single_note_object.get('date', '')
        tags = single_note_object.get('tags', [])

        # Create safe folder name with proper null handling
        safe_title = sanitize_filename(title) if title else 'Untitled'
        safe_date = sanitize_filename(date) if date else 'Unknown_Date'
        folder_name = f"{safe_title}_{safe_date}"

    else:
        logger.warning(f"Unparsable note detected: {uuid}")
        folder_name = os.path.splitext(os.path.basename(uuid))[
            0]  # Use the UUID as fallback
        # markdown_content = response_data["choices"][0]["message"]["content"]
        # todo: may need to break out of here or return early

    markdown_content = f"# {title}\n\n"

    if multi_note_upload:
        markdown_content += f"## Summary: {summary}"

    # document date, different from individual response dates, which are listed below for multi-note uploads
    markdown_content += f"**Date:** {date}\n\n"

    if tags:
        markdown_content += f"**Tags:** {' '.join([f'#{tag}' for tag in tags])}\n\n"

    for individual_response in individual_responses:

        individual_response_data = individual_response.get('transcription', '')

        if multi_note_upload:
            markdown_content += f"{individual_response_data.get('date','')}\n\n"

        markdown_content += f"{individual_response_data.get('transcription','')}\n\n"

    # Ensure unique folder name
    original_folder_name = folder_name
    counter = 2
    while os.path.exists(os.path.join(output_dir, folder_name)):
        folder_name = f"{original_folder_name}_{counter}"
        counter += 1

    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    markdown_file_path = os.path.join(folder_path, f"{folder_name}.md")
    with open(markdown_file_path, "w") as markdown_file:
        markdown_file.write(markdown_content)

    # Create an 'images' subfolder within the UUID folder
    images_folder_path = os.path.join(folder_path, "images")
    os.makedirs(images_folder_path, exist_ok=True)

    # Get image paths from response data
    image_paths = response_data.get("image_paths", [])

    logger.debug(f"Found {len(image_paths)} image paths for {folder_name}")
    logger.debug(f"Image paths: {image_paths}")

    # Copy all relevant images to the 'images' subfolder
    for image_path in image_paths:
        logger.debug(f'Processing image path: {image_path}')

        # Handle both absolute and relative paths
        if os.path.exists(image_path):
            try:
                shutil.copy2(image_path, images_folder_path)
                logger.info(
                    f'Successfully copied {os.path.basename(image_path)}')
            except Exception as e:
                logger.error(f'Failed to copy {image_path}: {e}')
        else:
            logger.warning(f'Image path does not exist: {image_path}')
            # Try without the ./ prefix if it exists
            clean_path = image_path.lstrip('./')
            if os.path.exists(clean_path):
                try:
                    shutil.copy2(clean_path, images_folder_path)
                    logger.info(
                        f'Successfully copied {os.path.basename(clean_path)} (cleaned path)')
                except Exception as e:
                    logger.error(
                        f'Failed to copy cleaned path {clean_path}: {e}')
            else:
                logger.error(
                    f'Neither original nor cleaned path exists: {image_path} -> {clean_path}')

logger.info("Each JSON element has its own folder in markdown_output.")
