#!/usr/bin/env python3
"""Convert JSON page data to Markdown files with YAML frontmatter.

This script reads JSON files from a json/ subfolder, extracts the
markdown content and metadata, and creates individual markdown files in
a markdown/ subfolder with YAML frontmatter containing URL, title, and description.

Usage:
    python convert_json_to_markdown.py <data_folder> [<data_folder2> ...]

Example:
    python convert_json_to_markdown.py data/help_hellotars_com data/docs_hellotars_com
"""

import json
import sys
from pathlib import Path


def url_to_filename(url: str) -> str:
    """Convert a URL to a valid filename by replacing special characters.

    Args:
        url: The URL to convert (e.g., "https://help.hellotars.com/en/...")

    Returns:
        A filename-safe string with :// and / replaced by underscores.
    """
    # Replace :// with _ and then / with _
    filename = url.replace("://", "_").replace("/", "_")
    # Remove trailing underscore if present
    if filename.endswith("_"):
        filename = filename[:-1]
    return f"{filename}.md"


def create_yaml_frontmatter(url: str, title: str, description: str) -> str:
    """Create YAML frontmatter block.

    Args:
        url: The page URL.
        title: The page title.
        description: The page description.

    Returns:
        A YAML frontmatter string enclosed in --- markers.
    """
    # Escape any quotes in the values
    title_escaped = title.replace('"', '\\"') if title else ""
    description_escaped = description.replace('"', '\\"') if description else ""

    return f'''---
url: "{url}"
title: "{title_escaped}"
description: "{description_escaped}"
---

'''


def process_json_file(json_path: Path, output_dir: Path) -> bool:
    """Process a single JSON file and create corresponding markdown file.

    Args:
        json_path: Path to the JSON file.
        output_dir: Directory to write the markdown file to.

    Returns:
        True if successful, False otherwise.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Extract fields
        markdown_content = data.get("markdown", "")
        metadata = data.get("metadata", {})
        url = metadata.get("url", "") or metadata.get("source_url", "")
        title = metadata.get("title", "")
        description = metadata.get("description", "")

        if not url:
            print(f"  Skipping {json_path.name}: No URL found")
            return False

        if not markdown_content:
            print(f"  Skipping {json_path.name}: No markdown content")
            return False

        # Create output filename
        filename = url_to_filename(url)
        output_path = output_dir / filename

        # Create markdown with frontmatter
        frontmatter = create_yaml_frontmatter(url, title, description)
        full_content = frontmatter + markdown_content

        # Write the file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        print(f"  Created: {filename}")
        return True

    except json.JSONDecodeError as e:
        print(f"  Error parsing {json_path.name}: {e}")
        return False
    except Exception as e:
        print(f"  Error processing {json_path.name}: {e}")
        return False


def process_data_folder(data_folder: Path) -> tuple[int, int]:
    """Process all JSON files in a data folder.

    Args:
        data_folder: Path to the data folder containing json/ subfolder.

    Returns:
        Tuple of (success_count, total_count).
    """
    input_dir = data_folder / "json"
    output_dir = data_folder / "markdown"

    # Verify input directory exists
    if not input_dir.exists():
        print(f"Error: JSON directory not found: {input_dir}")
        return 0, 0

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .json files in the json subfolder
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0, 0

    print(f"Found {len(json_files)} JSON files to process")
    print(f"Output directory: {output_dir}")
    print()

    success_count = 0
    for json_file in json_files:
        if process_json_file(json_file, output_dir):
            success_count += 1

    return success_count, len(json_files)


def main() -> None:
    """Main entry point for the script."""
    project_root = Path(__file__).parent.parent

    # Get data folders from command line arguments or use defaults
    if len(sys.argv) > 1:
        data_folders = [Path(arg) for arg in sys.argv[1:]]
        # Make paths absolute if they're relative
        data_folders = [
            project_root / folder if not folder.is_absolute() else folder
            for folder in data_folders
        ]
    else:
        # Default: process help_hellotars_com
        data_folders = [project_root / "data" / "help_hellotars_com"]

    total_success = 0
    total_files = 0

    for data_folder in data_folders:
        print(f"=" * 60)
        print(f"Processing: {data_folder.name}")
        print(f"=" * 60)

        success, total = process_data_folder(data_folder)
        total_success += success
        total_files += total

        print()
        print(f"Completed: {success}/{total} files converted successfully")
        print()

    if len(data_folders) > 1:
        print(f"=" * 60)
        print(f"TOTAL: {total_success}/{total_files} files converted successfully")
        print(f"=" * 60)


if __name__ == "__main__":
    main()
