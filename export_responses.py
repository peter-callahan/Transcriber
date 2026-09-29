import os
import re
import sys
import json
import shutil
import yaml
from datetime import datetime

from pipeline_utils import logger, parse_date_string

OUTPUT_FOLDER = os.path.expanduser(os.getenv('OUTPUT_FOLDER', './markdown_output'))
CURRENT_RESPONSES_FILE = os.getenv('CURRENT_RESPONSES_FILE', 'responses_current.json')
EXPORTABLE = ("done", "warning")
DOCUMENT_TYPE = "journal entry"
SOURCE = "digital-conversion"


def sanitize_filename(filename):
    if not filename:
        return 'Untitled'
    sanitized = re.sub(r'[/\\:*?"<>|]', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_ ')
    return sanitized or 'Untitled'


def _format_timestamp(dt):
    """YYYY_MM_DD HH:MM:SS +HH:MM (colon inserted into the UTC offset)."""
    s = dt.strftime('%Y_%m_%d %H:%M:%S %z')
    if len(s) >= 5 and s[-5] in '+-':
        s = f"{s[:-2]}:{s[-2:]}"
    return s


def _build_frontmatter(title, date_created, date_modified, tags):
    """YAML frontmatter block. Uses yaml.safe_dump throughout so a title or tag
    containing a colon/quote can never corrupt the block's structure."""
    scalars = {
        "title": title,
        "date created": date_created,
        "date modified": date_modified,
    }
    head = yaml.safe_dump(scalars, sort_keys=False, allow_unicode=True).rstrip("\n")
    tags_line = "tags: " + yaml.safe_dump(
        tags, default_flow_style=True, allow_unicode=True, default_style="'"
    ).rstrip("\n")
    tail = yaml.safe_dump(
        {"document_type": DOCUMENT_TYPE, "source": SOURCE}, sort_keys=False, allow_unicode=True
    ).rstrip("\n")
    return f"---\n{head}\n{tags_line}\n{tail}\n---\n\n"


def build_markdown(group, exported_at=None):
    """(folder_name, markdown) for one exportable group dict."""
    exported_at = exported_at or datetime.now().astimezone()
    summary = group.get("summary")
    pages = group.get("pages", [])
    if summary:
        title, date, tags = summary.get("title", ""), summary.get("date", ""), summary.get("tags", [])
    elif pages and pages[0].get("data"):
        data = pages[0]["data"]
        title, date, tags = data.get("title", ""), data.get("date", ""), data.get("tags", [])
    else:
        raise ValueError(f"group {group.get('group_name')} has no summary and no page data")

    # Mechanical (non-LLM) date normalization: coerces a partial date like "August 2020"
    # to 2020_08_01; leaves a full date untouched; falls back to the raw string if it
    # matches no known pattern at all, rather than discarding it.
    date = parse_date_string(date) or date

    folder_name = f"{sanitize_filename(date) if date else 'Unknown_Date'} - {sanitize_filename(title)}"

    md = _build_frontmatter(title, date, _format_timestamp(exported_at), tags)
    if summary:
        md += f"## Summary\n\n{summary.get('summary', '')}\n\n"

    if summary and summary.get("continuous_transcription"):
        md += f"{summary['continuous_transcription']}\n\n"
    else:
        for page in pages:
            data = page.get("data") or {}
            if summary and data.get("date"):
                md += f"{data['date']}\n\n"
            md += f"{data.get('transcription', '')}\n\n"
    return folder_name, md


def _export_group(group, output_dir):
    folder_name, md = build_markdown(group)
    candidate, counter = folder_name, 2
    while os.path.exists(os.path.join(output_dir, candidate)):
        candidate = f"{folder_name}_{counter}"
        counter += 1
    folder_path = os.path.join(output_dir, candidate)
    images_path = os.path.join(folder_path, "images")
    os.makedirs(images_path, exist_ok=True)

    with open(os.path.join(folder_path, f"{candidate}.md"), "w") as f:
        f.write(md)
    for image_path in group.get("image_paths", []):
        if os.path.exists(image_path):
            shutil.copy2(image_path, images_path)
        else:
            logger.warning(f"Image path does not exist: {image_path}")
    return folder_path


def export_run(results, output_dir):
    """Export done/warning groups. Returns one status dict per attempted group."""
    os.makedirs(output_dir, exist_ok=True)
    report = []
    for group in results:
        name = group.get("group_name")
        if group.get("status") not in EXPORTABLE:
            logger.info(f"Skipping export for {name} (status {group.get('status')})")
            continue
        try:
            path = _export_group(group, output_dir)
            logger.info(f"Exported {name} to {path}")
            report.append({"group_name": name, "ok": True, "path": path, "error": None})
        except Exception as e:
            logger.error(f"Export failed for {name}: {e}")
            report.append({"group_name": name, "ok": False, "path": None, "error": str(e)})
    return report


if __name__ == "__main__":
    if not os.path.exists(CURRENT_RESPONSES_FILE):
        logger.warning(f"No {CURRENT_RESPONSES_FILE} found — nothing to export.")
        sys.exit(0)
    with open(CURRENT_RESPONSES_FILE) as f:
        groups = list(json.load(f).values())
    report = export_run(groups, OUTPUT_FOLDER)
    sys.exit(0 if all(r["ok"] for r in report) else 1)
