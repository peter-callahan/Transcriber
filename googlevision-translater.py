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
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                if os.path.isfile(image_path) and image_file.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
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
