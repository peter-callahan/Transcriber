import os
import re
import sys
import json
import shutil

from pipeline_utils import logger

OUTPUT_FOLDER = os.path.expanduser(os.getenv('OUTPUT_FOLDER', './markdown_output'))
CURRENT_RESPONSES_FILE = os.getenv('CURRENT_RESPONSES_FILE', 'responses_current.json')
EXPORTABLE = ("done", "warning")


def sanitize_filename(filename):
    if not filename:
        return 'Untitled'
    sanitized = re.sub(r'[/\\:*?"<>|]', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_ ')
    return sanitized or 'Untitled'


def build_markdown(group):
    """(folder_name, markdown) for one exportable group dict."""
    summary = group.get("summary")
    pages = group.get("pages", [])
    if summary:
        title, date, tags = summary.get("title", ""), summary.get("date", ""), summary.get("tags", [])
    elif pages and pages[0].get("data"):
        data = pages[0]["data"]
        title, date, tags = data.get("title", ""), data.get("date", ""), data.get("tags", [])
    else:
        raise ValueError(f"group {group.get('group_name')} has no summary and no page data")

    folder_name = f"{sanitize_filename(date) if date else 'Unknown_Date'} - {sanitize_filename(title)}"

    md = f"# {title}\n\n"
    if summary:
        md += f"## Summary\n\n{summary.get('summary', '')}\n\n"
    md += f"**Date:** {date}\n\n"
    if tags:
        md += f"**Tags:** {' '.join(f'#{t}' for t in tags)}\n\n"

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
