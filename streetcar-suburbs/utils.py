"""Shared utilities for the Chicago Sun-Times article pipeline."""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


def strip_html(html_text: str) -> str:
    """Strip HTML tags from text, collapsing whitespace.

    Removes <img> and <figure> elements entirely (no alt text noise),
    then extracts remaining text content.
    """
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all(["img", "figure"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_json_files_from_dir(data_dir: Path) -> list[dict]:
    """Load all article entries from YYYY-MM-DD subdirectories.

    Walks data_dir looking for YYYY-MM-DD/YYYY-MM-DD.json files,
    reads each, and returns a flat list of article entry dicts with
    a _date field injected from the file's top-level date field.

    Args:
        data_dir: Path to the 2026 directory containing date subdirs.

    Returns:
        List of article entry dicts, sorted by date ascending.
    """
    articles = []
    date_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)]
    )
    for date_dir in date_dirs:
        json_file = date_dir / f"{date_dir.name}.json"
        if not json_file.exists():
            continue
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        date_str = data.get("date", date_dir.name)
        for entry in data.get("entries", []):
            entry["_date"] = date_str
            articles.append(entry)
    return articles
