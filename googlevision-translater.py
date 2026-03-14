import os
import sys
import json
import logging
from dotenv import load_dotenv
from google.cloud import vision_v1 as vision

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            logger.warning(f"Failed to read order.json: {e}, falling back to sorted order")

    # Fallback: return sorted list of image files
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]
    return sorted(image_files)


# Ensure Google credentials are set in environment
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

def extract_text_from_image(image_path, output_path, prompt=None):
    # Initialize the Google Vision client
    client = vision.ImageAnnotatorClient()

    # Read the image file
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    # Construct the request payload
    request = {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            }
        ]
    }

    # Perform the request
    response = client.annotate_image(request["requests"][0])

    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")

    # Extract blocks of text
    extracted_text = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            block_text = " ".join(
                [
                    "".join([symbol.text for symbol in word.symbols])
                    for paragraph in block.paragraphs
                    for word in paragraph.words
                ]
            )
            extracted_text.append(block_text)

    # Combine blocks into a single text
    combined_text = "\n\n".join(extracted_text)

    # Add the prompt if provided
    if prompt:
        combined_text = f"{prompt}\n\n{combined_text}"

    # Save the detected text to a file
    with open(output_path, "w") as output_file:
        output_file.write(combined_text)

    logger.info(f"Text extracted and saved to {output_path}")


# Load config with fallback
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    input_images_dir = os.path.expanduser(config['input_folder'])
except FileNotFoundError:
    # Default fallback when config.json doesn't exist
    input_images_dir = "input_images"

if __name__ == "__main__":
    # Check if specific group was provided as argument
    if len(sys.argv) > 1:
        group_name = sys.argv[1]
        # Process only the specified group folder
        folder_path = os.path.join(input_images_dir, group_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            logger.info(f"Processing OCR for group: {group_name}")
            file_order = get_file_order(folder_path)
            logger.info(f"Processing {len(file_order)} files in order: {file_order}")

            for image_file in file_order:
                image_path = os.path.join(folder_path, image_file)
                if os.path.isfile(image_path) and image_file.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
                    logger.info(f"Processing OCR for: {image_file}")
                    # Generate the output text file name
                    base_name, _ = os.path.splitext(image_file)
                    output_path = os.path.join(folder_path, f"{base_name}.txt")

                    # Extract text from the image and save it
                    extract_text_from_image(image_path, output_path, prompt=None)
        else:
            logger.warning(f"Group folder {group_name} not found")
            sys.exit(1)
    else:
        # Process all folders (original behavior)
        logger.info("Processing OCR for all groups")
        for folder_name in os.listdir(input_images_dir):
            folder_path = os.path.join(input_images_dir, folder_name)

            if os.path.isdir(folder_path):
                for image_file in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_file)
                    if os.path.isfile(image_path) and image_file.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
                        # Generate the output text file name
                        base_name, _ = os.path.splitext(image_file)
                        output_path = os.path.join(folder_path, f"{base_name}.txt")

                        # Extract text from the image and save it
                        extract_text_from_image(image_path, output_path, prompt=None)
