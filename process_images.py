import os
import sys
from PIL import Image
from pillow_heif import register_heif_opener

from pipeline_utils import (
    get_file_order, IMAGE_EXTENSIONS, INPUT_FOLDER, StageResult, logger,
)

register_heif_opener()


def resize_image(image_path, max_size=(2000, 2000)):
    """Shrink to max_size and convert to JPEG in place. Returns the output path."""
    logger.info(f"Processing image: {image_path}")
    with Image.open(image_path) as img:
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.load()
        if img.format != "JPEG":
            output_path = os.path.splitext(image_path)[0] + ".jpg"
            logger.info(f"Converting {img.format} to JPEG: {image_path} -> {output_path}")
        else:
            output_path = image_path
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=85)

    if output_path != image_path and os.path.exists(image_path):
        os.remove(image_path)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Failed to save {output_path}")
    logger.info(f"Successfully processed: {output_path}")
    return output_path


def process_group(group_name, input_dir=None):
    folder = os.path.join(input_dir or INPUT_FOLDER, group_name)
    if not os.path.isdir(folder):
        return StageResult(False, f"Group folder not found: {folder}")

    errors = []
    file_order = get_file_order(folder)
    logger.info(f"Resizing {len(file_order)} files in group {group_name}: {file_order}")
    for image_file in file_order:
        image_path = os.path.join(folder, image_file)
        if not (os.path.isfile(image_path) and image_file.lower().endswith(IMAGE_EXTENSIONS)):
            continue
        try:
            resize_image(image_path)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            errors.append(f"{image_file}: {e}")

    if errors:
        return StageResult(False, "; ".join(errors))
    return StageResult(True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = process_group(sys.argv[1])
        if not result.ok:
            logger.error(result.error)
            sys.exit(1)
    else:
        for name in sorted(os.listdir(INPUT_FOLDER)):
            if os.path.isdir(os.path.join(INPUT_FOLDER, name)):
                process_group(name)
