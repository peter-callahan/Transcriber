import os
import sys
from google.cloud import vision_v1 as vision

from pipeline_utils import (
    get_file_order, resolve_image_path, IMAGE_EXTENSIONS, INPUT_FOLDER, StageResult, logger,
)


def extract_text_from_image(image_path):
    """Document text detection via Google Vision. Returns the block-joined text."""
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as f:
        content = f.read()
    response = client.annotate_image({
        "image": {"content": content},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
    })
    if response.error.message:
        raise RuntimeError(f"Google Vision API error: {response.error.message}")

    blocks = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            blocks.append(" ".join(
                "".join(symbol.text for symbol in word.symbols)
                for paragraph in block.paragraphs
                for word in paragraph.words
            ))
    text = "\n\n".join(blocks)
    logger.info(f"OCR content ({len(text)} chars): {text[:200]}{'...' if len(text) > 200 else ''}")
    return text


def process_group(group_name, input_dir=None, extract_fn=None):
    extract_fn = extract_fn or extract_text_from_image
    folder = os.path.join(input_dir or INPUT_FOLDER, group_name)
    if not os.path.isdir(folder):
        return StageResult(False, f"Group folder not found: {folder}")

    errors = []
    file_order = get_file_order(folder)
    logger.info(f"OCR for {len(file_order)} files in group {group_name}: {file_order}")
    for image_file in file_order:
        resolved = resolve_image_path(folder, image_file)
        if resolved is None:
            logger.warning(f"Skipping OCR for {image_file} — file not found")
            continue
        name, path = resolved
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        try:
            text = extract_fn(path)
        except Exception as e:
            logger.error(f"OCR failed for {name}: {e}")
            errors.append(f"{name}: {e}")
            continue
        output_path = os.path.join(folder, os.path.splitext(name)[0] + ".txt")
        with open(output_path, "w") as f:
            f.write(text)
        logger.info(f"Text extracted and saved to {output_path}")

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
