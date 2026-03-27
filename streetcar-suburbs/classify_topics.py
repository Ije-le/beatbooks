"""Classify streetcar suburbs articles by topic.

Reads articles from streetcar-suburbs/streetcarsuburbs.json — either from a
local copy or directly from the NewsAppsUMD/beat_book_work GitHub repository.
Uses an LLM to assign one or more topics to each article. Outputs a separate
JSON file per topic.

Topics:
    city_council, public_safety, education, arts_culture,
    food, business, security, other

Usage:
    # Fetch articles automatically from GitHub (default):
    uv run python classify_topics.py --model groq-llama-3.3-70b

    # Or point at a local file:
    uv run python classify_topics.py \\
        --model groq-llama-3.3-70b \\
        --data-file streetcar-suburbs/streetcarsuburbs.json
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

TOPICS = [
    "city_council",
    "public_safety",
    "education",
    "arts_culture",
    "food",
    "business",
    "security",
    "sports",
    "other",
]

SYSTEM_PROMPT = """You are a news article classifier. You only classify articles that are
specifically about College Park, Maryland. If the article is not about College Park,
Maryland, return only: other

If the article IS about College Park, Maryland, assign one or more topics from this list:

- city_council: The article explicitly involves the College Park city council, mayor,
  or city government — city votes, ordinances, city budget, or city policy decisions.
  Must be College Park CITY government, not Prince George's County or the state.

- public_safety: The article covers a specific fire, accident, public health emergency,
  or community safety incident in College Park. Do NOT use this for general crime or
  policing — that is security.

- education: The article is specifically about a school, school board decision, classroom,
  teacher, student achievement, or education policy in College Park. Do NOT use this for
  cultural events, celebrations, or programs that merely take place at a school.

- arts_culture: The article is specifically about an art exhibition, musical performance,
  theater production, film, cultural festival, museum, or community celebration in College
  Park. Lunar New Year, cultural heritage events, and community arts programs belong here.

- food: The article is specifically about a restaurant, food business, food policy, or
  food-focused community event in College Park. Must have a clear food angle.

- business: The article is specifically about a local business opening, closing, economic
  development project, jobs, or commerce in College Park. Do NOT use for stories that
  merely mention a business in passing.

- security: The article is specifically about a crime, arrest, police action, court case,
  or criminal justice matter in College Park.

- sports: The article is specifically about a sports team, athlete, game, tournament,
  or recreational sports program in College Park.

- other: The article is about College Park but does not clearly fit any of the above.
  Do NOT assign other if the article already fits one or more of the above topics.

An article can belong to multiple topics. Return ONLY a comma-separated list of topic
names from the list above, nothing else. Example: city_council,education"""

USER_PROMPT_TEMPLATE = """Article title: {title}

Article content:
{content}

Is this article about College Park, Maryland? If not, return: other
If yes, which topics apply? Return only a comma-separated list from:
city_council, public_safety, education, arts_culture, food, business, security, sports, other
Only use "other" if the article does not fit any of the other topics."""


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


def classify_article(model, article: dict) -> list[str]:
    """Return a list of topics for the article."""
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
    raw_answer = response.text().strip().lower()

    # Parse the comma-separated response and validate against known topics
    assigned = [t.strip() for t in raw_answer.split(",") if t.strip() in TOPICS]
    # Only fall back to "other" if nothing else was assigned
    specific = [t for t in assigned if t != "other"]
    if not specific:
        assigned = ["other"]
    else:
        assigned = specific
    return assigned


def main():
    parser = argparse.ArgumentParser(
        description="Classify streetcar suburbs articles by topic."
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
        help="Model name to use for classification (e.g. groq-llama-3.3-70b).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        type=Path,
        help="Directory to write topic JSON files (default: current directory).",
    )
    parser.add_argument(
        "--state-file",
        default=".classify_topics_state.json",
        type=Path,
        help="State file for resuming interrupted runs (default: .classify_topics_state.json).",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Only process the first N articles (useful for testing).",
    )
    args = parser.parse_args()

    console.print("[bold]Streetcar Suburbs Topic Classifier[/bold]")
    console.print(f"  Model      : {args.model}")
    console.print(f"  Output dir : {args.output_dir}")
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

    # Load state
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
    topic_buckets: dict[str, list[dict]] = {t: [] for t in TOPICS}

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
                for topic in state[article_id]:
                    if topic in topic_buckets:
                        topic_buckets[topic].append(article)
                progress.advance(task)
                continue

            topics = classify_article(model, article)
            state[article_id] = topics
            save_state(state, args.state_file)

            for topic in topics:
                topic_buckets[topic].append(article)

            progress.advance(task)

    console.print()
    console.print("[bold green]Classification complete.[/bold green]")
    for topic, bucket in topic_buckets.items():
        console.print(f"  {topic:<15}: {len(bucket)}")

    # Write one output file per topic
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_test" if args.limit is not None else ""
    for topic, bucket in topic_buckets.items():
        out_path = args.output_dir / f"{topic}{suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"topic": topic, "total": len(bucket), "model": args.model, "articles": bucket}, f, indent=2, ensure_ascii=False)
        console.print(f"  Wrote {len(bucket):>4} articles to [bold]{out_path}[/bold]")


if __name__ == "__main__":
    main()
