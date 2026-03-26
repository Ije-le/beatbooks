"""Classify streetcar suburbs articles as College Park City Council beat stories.

Reads articles from a single local JSON file or directly from the
NewsAppsUMD/beat_book_work GitHub repository. Uses an LLM to determine
whether each article covers the College Park City Council beat. Outputs
a filtered JSON file.

Usage:
    # Pull articles automatically from GitHub (default):
    uv run python classify_streetcar.py --model groq/meta-llama/llama-4-scout-17b-16e-instruct

    # Or point at a local file:
    uv run python classify_streetcar.py \\
        --data-file /path/to/streetcarsuburbs.json \\
        --model llama3.2 \\
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

SYSTEM_PROMPT = """You are a news article classifier for the College Park City Council beat.

Your job is to determine if an article covers the COLLEGE PARK CITY COUNCIL BEAT —
meaning it involves the College Park, Maryland city council, its members, meetings,
votes, decisions, or the local legislative and governance process. Answer only YES or NO.

Topics that qualify as College Park City Council beat:
- College Park City Council meetings, agendas, votes, and resolutions
- Council member activity, elections, appointments, or resignations
- Mayor of College Park and their relationship with the council
- Local ordinances and municipal code changes affecting College Park
- Zoning, land-use, and development decisions by the College Park council or planning board
- College Park city budget, taxes, and spending approved by the council
- Council oversight of College Park city departments and services
- University of Maryland and its relationship with College Park city government
- Public hearings and community input processes held by the council
- Intergovernmental agreements involving College Park city government
- College Park infrastructure, public works, or transit decisions by the council

Topics that do NOT qualify:
- University of Maryland campus news with no city council angle
- Prince George's County government actions (unless College Park council is also involved)
- State or federal legislation not directly tied to a College Park council decision
- General crime stories without a council policy angle
- School board or park authority decisions (separate governing bodies)
- Human interest stories without a College Park governance angle
- Business news without a College Park city council decision involved

Be conservative: if unclear, answer NO."""

USER_PROMPT_TEMPLATE = """Article title: {title}

Article content:
{summary}

Does this article cover the College Park City Council beat — meaning it involves the
College Park, Maryland city council and their meetings, votes, members, or decisions?
Answer only YES or NO."""


GITHUB_REPO = "NewsAppsUMD/beat_book_work"
GITHUB_DATA_FILE = "streetcar-suburbs/streetcarsuburbs.json"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/{repo}/main/{path}"


def load_articles_from_github() -> list[dict]:
    """Fetch articles from the single streetcarsuburbs.json file in the GitHub repo."""
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
    """Return True if the article belongs to the city council beat."""
    title = article.get("title", "")

    # Prefer the AI-generated summary from extraction; fall back to raw summary field
    extraction = article.get("extraction", {})
    raw_summary = extraction.get("ai_summary") or article.get("summary", "")
    summary_text = strip_html(raw_summary)[:2000]

    prompt = USER_PROMPT_TEMPLATE.format(title=title, summary=summary_text)
    try:
        response = model.prompt(prompt, system=SYSTEM_PROMPT)
        answer = response.text().strip().upper()
        return answer.startswith("YES")
    except Exception as exc:
        console.print(f"[yellow]Warning: classification error for '{title[:60]}': {exc}[/yellow]")
        return False


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
            "If omitted, the file is fetched automatically from the "
            "NewsAppsUMD/beat_book_work GitHub repository."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model name to use for classification (e.g. llama3.2, groq/...).",
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

    # Load articles
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

    # Load model
    try:
        model = llm.get_model(args.model)
    except Exception as exc:
        console.print(f"[red]Error loading model '{args.model}': {exc}[/red]")
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
