"""Classify streetcar suburbs articles as College Park City Council beat stories.

Reads articles from streetcar-suburbs/streetcarsuburbs.json — either from a
local copy or directly from the NewsAppsUMD/beat_book_work GitHub repository.
Uses an Ollama model to determine whether each article covers the College Park
City Council beat. Outputs a filtered JSON file.

Usage:
    # Fetch articles automatically from GitHub (default):
    uv run python classify_streetcar.py --model llama3.2

    # Or point at a local file:
    uv run python classify_streetcar.py \\
        --model llama3.2 \\
        --data-file streetcar-suburbs/streetcarsuburbs.json \\
        [--output classified_streetcar.json] \\
        [--state-file .classify_streetcar_state.json]
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import llm
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from utils import strip_html

console = Console()

GITHUB_REPO = "NewsAppsUMD/beat_book_work"
GITHUB_DATA_FILE = "streetcar-suburbs/streetcarsuburbs.json"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/{repo}/main/{path}"

SYSTEM_PROMPT = """You are a news article classifier for the College Park city government beat.

Your job is to determine if an article involves the city government of College Park,
Maryland specifically — not Prince George's County, not the state of Maryland, not
other municipalities. Answer only YES or NO.

Say YES if the article explicitly involves College Park city government, such as:
- The College Park City Council or College Park city government
- A College Park council member, alderman, or the College Park mayor
- A vote, resolution, ordinance, or meeting of the College Park city government
- College Park city budget, taxes, spending, or city services
- Zoning or development decisions made by College Park city officials
- College Park city departments or city staff
- The University of Maryland's relationship specifically with College Park city government

Say NO if:
- The article is only about Prince George's County government or the county council
- The article is about state government without a College Park city angle
- Council members mentioned are county or state officials, not College Park city officials
- College Park is only mentioned geographically, not in a city government context
- The article covers other suburbs or municipalities without involving College Park city government

The article must clearly involve College Park CITY government specifically to answer YES."""

USER_PROMPT_TEMPLATE = """Article title: {title}

Article content:
{content}

Does this article specifically involve the city government of College Park, Maryland —
not the county, not the state, but College Park city officials, council, or city policy?
Answer only YES or NO."""


def load_articles_from_github() -> list[dict]:
    """Fetch articles from streetcarsuburbs.json in the GitHub repo."""
    url = GITHUB_RAW_BASE.format(repo=GITHUB_REPO, path=GITHUB_DATA_FILE)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return data.get("articles", data) if isinstance(data, dict) else data


def load_articles_from_file(file_path: Path) -> list[dict]:
    """Load articles from a local JSON file."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("articles", data) if isinstance(data, dict) else data


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict, state_file: Path) -> None:
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f)


def classify_article(model, article: dict) -> bool:
    """Return True if the article belongs to the College Park City Council beat."""
    raw_title = article.get("title", "")
    if isinstance(raw_title, dict):
        title = raw_title.get("rendered", str(raw_title))
    else:
        title = str(raw_title)

    raw_content = article.get("content", "")
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)
    content_text = strip_html(raw_content)[:2000]

    prompt = USER_PROMPT_TEMPLATE.format(title=title, content=content_text)
    response = model.prompt(prompt, system=SYSTEM_PROMPT)
    answer = response.text().strip().upper()
    return answer.startswith("YES")


def main():
    parser = argparse.ArgumentParser(
        description="Classify streetcar suburbs articles as College Park City Council beat stories."
    )
    parser.add_argument(
        "--data-file",
        default=None,
        type=Path,
        help=(
            "Path to a local streetcarsuburbs.json file. "
            "If omitted, articles are fetched from GitHub automatically."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name to use for classification (e.g. llama3.2, mistral).",
    )
    parser.add_argument(
        "--output",
        default="classified_streetcar.json",
        type=Path,
        help="Output JSON file path (default: classified_streetcar.json).",
    )
    parser.add_argument(
        "--state-file",
        default=".classify_streetcar_state.json",
        type=Path,
        help="State file for resuming interrupted runs (default: .classify_streetcar_state.json).",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Only process the first N articles (useful for testing).",
    )
    args = parser.parse_args()

    if args.limit is not None:
        args.output = args.output.with_stem(args.output.stem + "_test")

    console.print("[bold]College Park City Council Beat Classifier[/bold]")
    console.print(f"  Model  : {args.model}")
    console.print(f"  Output : {args.output}")
    console.print()

    if args.data_file is not None:
        if not args.data_file.exists():
            console.print(f"[red]Error: file not found: {args.data_file}[/red]")
            sys.exit(1)
        console.print(f"  Source : {args.data_file}")
        articles = load_articles_from_file(args.data_file)
    else:
        console.print(f"  Source : GitHub ({GITHUB_REPO} — {GITHUB_DATA_FILE})")
        console.print("Fetching articles from GitHub...")
        try:
            articles = load_articles_from_github()
        except Exception as exc:
            console.print(f"[red]Error fetching articles from GitHub: {exc}[/red]")
            sys.exit(1)
    console.print(f"  Found {len(articles)} articles.")

    if args.limit is not None:
        articles = articles[: args.limit]
        console.print(f"  [cyan]Limiting to first {len(articles)} articles (--limit).[/cyan]")
    console.print()

    # Load state (already-classified article IDs)
    state = load_state(args.state_file)
    already_done = sum(1 for v in state.values() if v is not None)
    if already_done:
        console.print(f"  Resuming: {already_done} articles already classified, skipping.")

    # Load Ollama model
    try:
        model = llm.get_model(args.model)
    except Exception as exc:
        console.print(f"[red]Error loading model '{args.model}': {exc}[/red]")
        console.print("  Make sure the model is pulled via Ollama and llm-ollama is installed.")
        sys.exit(1)

    # Classify articles
    council_articles = []
    skipped = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Classifying...", total=len(articles))

        for article in articles:
            article_id = article.get("id") or article.get("link", "")

            if article_id in state:
                if state[article_id]:
                    council_articles.append(article)
                skipped += 1
                progress.advance(task)
                continue

            is_council = classify_article(model, article)
            state[article_id] = is_council
            save_state(state, args.state_file)

            if is_council:
                council_articles.append(article)

            progress.advance(task)

    console.print()
    console.print("[bold green]Classification complete.[/bold green]")
    console.print(f"  Total input    : {len(articles)}")
    console.print(f"  College Park CC: {len(council_articles)}")
    console.print(f"  Filtered out   : {len(articles) - len(council_articles)}")
    if skipped:
        console.print(f"  Skipped (cached): {skipped}")

    # Write output
    output = {
        "total_input": len(articles),
        "total_classified": len(council_articles),
        "model": args.model,
        "articles": council_articles,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    console.print(f"\nWrote {len(council_articles)} articles to [bold]{args.output}[/bold]")


if __name__ == "__main__":
    main()
