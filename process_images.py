import os
import json
import sys
import logging
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF/HEIC format support
register_heif_opener()

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


def resize_image(image_path, max_size=(2048, 2048)):
    try:
        # Log what we're attempting to process
        logger.info(f"Processing image: {image_path}")
        if not os.path.exists(image_path):
            logger.error(f"File does not exist: {image_path}")
            return

        file_size = os.path.getsize(image_path)
        logger.debug(f"File size: {file_size} bytes")

        # Open, process, and load image data into memory
        with Image.open(image_path) as img:
            logger.debug(f"Image format: {img.format}, size: {img.size}, mode: {img.mode}")
            # Only resize if image is actually larger than max_size
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)  # High quality resampling

            # Load image data to avoid issues with saving to same path
            img.load()

            # Determine output path
            if img.format != "JPEG":
                output_path = os.path.splitext(image_path)[0] + ".jpg"
                should_remove_original = (output_path != image_path)  # Only remove if different
                logger.info(f"Converting {img.format} to JPEG: {image_path} -> {output_path}")
            else:
                output_path = image_path
                should_remove_original = False
                logger.debug(f"Image already JPEG, saving in place")

            # Save the image (context manager is still open, but img.load() makes it safe)
            img.save(output_path, "JPEG", quality=95)
            logger.debug(f"Saved to {output_path}")

        # Now that the context manager has closed, safe to remove original if needed
        # Important: only remove if output path is different from input path
        if should_remove_original and os.path.exists(image_path) and output_path != image_path:
            logger.debug(f"Removing original file: {image_path}")
            os.remove(image_path)

        # Verify the output file exists
        if not os.path.exists(output_path):
            logger.error(f"Output file does not exist after save: {output_path}")
            raise FileNotFoundError(f"Failed to save {output_path}")

        logger.info(f"Successfully processed: {output_path}")
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        # Try to convert HEIC using a different method if available
        if image_path.lower().endswith('.heic'):
            try:
                # Re-register and try again
                from pillow_heif import register_heif_opener
                register_heif_opener()
                with Image.open(image_path) as img:
                    # Only resize if image is actually larger than max_size
                    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                        img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    new_image_path = os.path.splitext(image_path)[0] + ".jpg"
                    img.save(new_image_path, "JPEG", quality=95)
                    os.remove(image_path)
                logger.info(f"Successfully processed HEIC on retry: {image_path}")
            except Exception as e2:
                logger.error(f"Failed to process HEIC file {image_path}: {e2}")
                # Skip this file rather than crashing
                return


# Load config with fallback
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    input_folder = os.path.expanduser(config['input_folder'])
except FileNotFoundError:
    # Default fallback when config.json doesn't exist
    input_folder = "input_images"

# Check if specific group was provided as argument
if len(sys.argv) > 1:
    group_name = sys.argv[1]
    # Process only the specified group folder
    group_folder = os.path.join(input_folder, group_name)
    if os.path.exists(group_folder):
        logger.info(f"Processing images in group: {group_name}")
        file_order = get_file_order(group_folder)
        logger.info(f"Processing {len(file_order)} files in order: {file_order}")

        for image_file in file_order:
            image_path = os.path.join(group_folder, image_file)
            if os.path.isfile(image_path) and image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                resize_image(image_path)
    else:
        logger.warning(f"Group folder {group_name} not found")
        sys.exit(1)
else:
    # Process all groups (original behavior)
    logger.info("Processing all images in input folder")
    for root, _, files in os.walk(input_folder):
        for image_file in files:
            image_path = os.path.join(root, image_file)
            if os.path.isfile(image_path) and image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                resize_image(image_path)
