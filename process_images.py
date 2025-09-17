import os
import json
import sys
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF/HEIC format support
register_heif_opener()


def resize_image(image_path, max_size=(2048, 2048)):
    try:
        with Image.open(image_path) as img:
            # Only resize if image is actually larger than max_size
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)  # High quality resampling
            # Ensure the image is saved as JPEG regardless of the input format
            if img.format != "JPEG":
                new_image_path = os.path.splitext(image_path)[0] + ".jpg"
                img.save(new_image_path, "JPEG", quality=95)  # High quality
                os.remove(image_path)  # Delete the original file
            else:
                img.save(image_path, "JPEG", quality=95)  # High quality
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
                    # Only resize if image is actually larger than max_size
                    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                        img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    new_image_path = os.path.splitext(image_path)[0] + ".jpg"
                    img.save(new_image_path, "JPEG", quality=95)
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

# Check if specific group was provided as argument
if len(sys.argv) > 1:
    group_name = sys.argv[1]
    # Process only the specified group folder
    group_folder = os.path.join(input_folder, group_name)
    if os.path.exists(group_folder):
        print(f"Processing images in group: {group_name}")
        for image_file in os.listdir(group_folder):
            image_path = os.path.join(group_folder, image_file)
            if os.path.isfile(image_path) and image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                resize_image(image_path)
    else:
        print(f"Group folder {group_name} not found")
        sys.exit(1)
else:
    # Process all groups (original behavior)
    print("Processing all images in input folder")
    for root, _, files in os.walk(input_folder):
        for image_file in files:
            image_path = os.path.join(root, image_file)
            if os.path.isfile(image_path) and image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                resize_image(image_path)
