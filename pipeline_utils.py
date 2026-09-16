import os
import json
import logging
from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict
from typing import Optional

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


@dataclass
class StageResult:
    ok: bool
    error: Optional[str] = None


@dataclass
class CallMeta:
    run_id: str
    group_name: str
    filename: Optional[str]
    kind: str                       # "page" | "summary"
    provider: str
    model: str
    prompt_hash: str
    attempts: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    word_count: int
    uncertain_count: int
    timestamp: str


@dataclass
class PageResult:
    filename: str
    status: str                     # "done" | "warning" | "failed"
    attempts: int = 0
    data: Optional[dict] = None
    uncertain: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    error: Optional[str] = None
    history: list = field(default_factory=list)
    meta: Optional[CallMeta] = None


@dataclass
class GroupResult:
    group_name: str
    status: str                     # "done" | "warning" | "failed"
    file_order: list
    image_paths: list
    pages: list                     # list[PageResult]
    summary: Optional[dict] = None
    summary_attempts: int = 0
    summary_meta: Optional[CallMeta] = None
    summary_error: Optional[str] = None
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def rollup_status(page_statuses, summary_error, group_warnings):
    if not page_statuses or "failed" in page_statuses or summary_error:
        return "failed"
    if "warning" in page_statuses or group_warnings:
        return "warning"
    return "done"
