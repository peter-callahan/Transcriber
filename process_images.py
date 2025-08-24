import os
import json
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF/HEIC format support
register_heif_opener()


def resize_image(image_path, max_size=(1024, 1024)):
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            # Ensure the image is saved as JPEG regardless of the input format
            if img.format != "JPEG":
                new_image_path = os.path.splitext(image_path)[0] + ".jpg"
                img.save(new_image_path, "JPEG", quality=85)
                os.remove(image_path)  # Delete the original file
            else:
                img.save(image_path, "JPEG", quality=85)
        print(f"Successfully processed: {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        # Try to convert HEIC using a different method if available
        if image_path.lower().endswith('.heic'):
            try:
                # Re-register and try again
                from pillow_heif import register_heif_opener
                register_heif_opener()
                with Image.open(image_path) as img:
                    img.thumbnail(max_size)
                    new_image_path = os.path.splitext(image_path)[0] + ".jpg"
                    img.save(new_image_path, "JPEG", quality=85)
                    os.remove(image_path)
                print(f"Successfully processed HEIC on retry: {image_path}")
            except Exception as e2:
                print(f"Failed to process HEIC file {image_path}: {e2}")
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

for root, _, files in os.walk(input_folder):
    for image_file in files:
        image_path = os.path.join(root, image_file)

        if os.path.isfile(image_path) and image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
            resize_image(image_path)
