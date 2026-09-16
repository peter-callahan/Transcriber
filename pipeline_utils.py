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
