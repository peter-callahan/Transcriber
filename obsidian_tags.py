#!/usr/bin/env python3
"""
Simple Obsidian hashtag extractor.
Scans a vault for #hashtags and returns them as a sorted list.
"""

import os
import re
import json
import logging
from pathlib import Path
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_obsidian_tags(vault_path):
    """Extract hashtags from Obsidian vault with frequency counts."""
    if not vault_path or not os.path.exists(vault_path):
        logger.error(f"Vault path not found: {vault_path}")
        return Counter()

    tag_counter = Counter()
    file_count = 0

    # Find all markdown files
    for md_file in Path(vault_path).rglob("*.md"):
        if '.obsidian' in md_file.parts:
            continue

        try:
            content = md_file.read_text(encoding='utf-8')

            # Remove HTML tags and their content to avoid false positives
            content = re.sub(r'<[^>]+>', '', content)

            # Remove URLs to avoid hashtag-like patterns in URLs
            content = re.sub(r'https?://[^\s]+', '', content)

            # Split content into lines for better context checking
            lines = content.split('\n')

            for line in lines:
                # Skip code blocks (lines starting with 4+ spaces or inside ```)
                if line.strip().startswith('    ') or '```' in line:
                    continue

                # Find potential hashtags
                potential_tags = re.findall(
                    r'(?<!#)#([a-zA-Z][a-zA-Z0-9_/-]*)', line)

                for tag in potential_tags:
                    # Filter out obvious false positives
                    if is_valid_tag(tag):
                        tag_counter[tag] += 1

            file_count += 1
        except:
            continue

    logger.info(f"Found {len(tag_counter)} unique hashtags in {file_count} files")
    return tag_counter


def is_valid_tag(tag):
    """Check if a tag looks like a real Obsidian tag vs. noise."""
    # Skip very long tags (likely URLs or IDs)
    if len(tag) > 30:
        return False

    # Skip tags that are mostly numbers (like IDs)
    if len(re.findall(r'\d', tag)) > len(tag) * 0.7:
        return False

    # Skip tags with too many consecutive uppercase letters (likely codes)
    if re.search(r'[A-Z]{4,}', tag):
        return False

    # Skip tags that look like file extensions or technical codes
    if re.match(r'^[a-zA-Z0-9]{10,}$', tag):  # Long alphanumeric strings
        return False

    # Skip single characters or very short tags that are likely noise
    if len(tag) < 2:
        return False

    return True


def save_tags(tag_counter, filename='obsidian_tags.json'):
    """Save tags to JSON file for use by other scripts."""
    # Convert Counter to dict for JSON serialization
    tag_dict = dict(tag_counter)
    with open(filename, 'w') as f:
        json.dump(tag_dict, f, indent=2)
    logger.info(f"Tags saved to {filename}")


def load_saved_tags(filename='obsidian_tags.json'):
    """Load previously saved tags from JSON file."""
    try:
        with open(filename, 'r') as f:
            tag_dict = json.load(f)
        logger.info(f"Loaded {len(tag_dict)} tags from {filename}")
        # Return just the tag names (keys) as a list for the main script
        return list(tag_dict.keys())
    except FileNotFoundError:
        logger.warning(f"No saved tags found at {filename}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error reading {filename}")
        return []


if __name__ == "__main__":
    # Example usage - set your vault path here
    VAULT_PATH = "/path/to/your/obsidian/vault"

    # You can also pass the path as a command line argument
    import sys
    if len(sys.argv) > 1:
        VAULT_PATH = sys.argv[1]

    logger.info(f"Scanning vault: {VAULT_PATH}")
    tags = get_obsidian_tags(VAULT_PATH)

    if tags:
        logger.info("\nMost common hashtags:")
        for tag, count in tags.most_common(50):
            logger.info(f"{count:3d}x #{tag}")

        logger.info("\nTags used only once (potential noise):")
        rare_tags = [tag for tag, count in tags.items() if count == 1]
        for tag in sorted(rare_tags)[:20]:  # Show first 20
            logger.info(f"  1x #{tag}")
        if len(rare_tags) > 20:
            logger.info(f"  ... and {len(rare_tags) - 20} more")

        # Save to both formats
        save_tags(tags, 'obsidian_tags.json')

        # Save just the frequent tags (used 2+ times) for the main script
        frequent_tags = [tag for tag, count in tags.items() if count >= 2]
        with open('obsidian_hashtags.txt', 'w') as f:
            for tag in sorted(frequent_tags):
                f.write(f"#{tag}\n")
        logger.info(f"\n{len(frequent_tags)} frequent hashtags saved to obsidian_hashtags.txt")
        logger.info(f"Filtered out {len(rare_tags)} single-use tags as potential noise")
    else:
        logger.warning("No hashtags found!")
