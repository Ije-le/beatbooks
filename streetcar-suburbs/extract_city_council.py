"""Extract structured metadata and summaries from city council articles.

Reads city_council.json, runs each article through a model to extract
key people, organizations, locations, issues, and a summary. Outputs
extracted_city_council.json.

Usage:
    uv run python extract_city_council.py \\
        --model groq-llama-3.3-70b \\
        [--input city_council.json] \\
        [--output extracted_city_council.json] \\
        [--state-file .extract_city_council_state.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import llm
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from utils import strip_html

console = Console()

CATEGORIES = [
    "zoning",
    "budget",
    "ordinance",
    "infrastructure",
    "education",
    "public_safety",
    "housing",
    "transportation",
    "environment",
    "community",
    "economic_development",
    "arts_culture",
    "health",
    "other",
]

SYSTEM_PROMPT = """You are a structured data extractor for College Park, Maryland city council news.

Given a news article, extract key information and return ONLY a valid JSON object.
Do not include markdown code fences, backticks, or any text outside the JSON object.
Use null for fields where information is not present in the article.
For the category field, choose the single best match from:
zoning, budget, ordinance, infrastructure, education, public_safety, housing,
transportation, environment, community, economic_development, arts_culture, health, other

Category definitions:
- zoning: land use, development, building approvals, permits
- budget: city finances, spending, tax levies, appropriations
- ordinance: new laws, code changes, resolutions, policy changes
- infrastructure: roads, public works, utilities, city services
- education: schools, University of Maryland relations, curriculum policy, campus-city issues
- public_safety: police, fire, emergency services, crime policy
- housing: affordable housing, rent, residential development
- transportation: transit, parking, bike lanes, traffic
- environment: sustainability, green initiatives, parks, trees
- community: neighborhood events, community programs, civic engagement
- economic_development: business attraction, commerce, jobs
- arts_culture: cultural programs, festivals, public art
- health: public health, city health programs, wellness
- other: does not fit any of the above"""

USER_PROMPT_TEMPLATE = """Extract structured metadata from this College Park city council article.

Title: {title}
URL: {url}
Published: {published}
Author: {author}

Content:
{content}

Return this exact JSON structure with values filled in:
{{
  "key_people": ["list of names each formatted as 'Full Name, Title/Role' — always include the person's title or role, e.g. 'Ahmed Andrew, College Park City Councilmember' or 'Anita Johnson, Mayor of College Park'"],
  "organizations": ["list of government agencies, departments, or institutions"],
  "locations": ["list of specific College Park neighborhoods, streets, or facilities"],
  "key_issues": ["list of 2-5 main policy issues or topics covered"],
  "category": "one of: zoning, budget, ordinance, infrastructure, education, public_safety, housing, transportation, environment, community, economic_development, arts_culture, health, other",
  "ai_summary": "2-3 sentence factual summary of what happened, who is involved, why it matters for College Park governance, and keep all quotes. Details must be accurate and verifiable from the article text. Do not add any information that is not explicitly stated in the article."
}}"""

DEFAULT_EXTRACTION = {
    "key_people": [],
    "organizations": [],
    "locations": [],
    "key_issues": [],
    "category": "other",
    "ai_summary": "",
    "_extraction_failed": True,
}


def parse_json_response(raw: str) -> dict | None:
    """Try to extract a JSON object from the model's response."""
    text = raw.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: markdown fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: first brace-delimited block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_extraction(data: dict) -> dict:
    """Ensure all expected fields exist and are the right types."""
    validated = {}
    validated["key_people"] = data.get("key_people") if isinstance(data.get("key_people"), list) else []
    validated["organizations"] = data.get("organizations") if isinstance(data.get("organizations"), list) else []
    validated["locations"] = data.get("locations") if isinstance(data.get("locations"), list) else []
    validated["key_issues"] = data.get("key_issues") if isinstance(data.get("key_issues"), list) else []
    cat = data.get("category", "other")
    validated["category"] = cat if cat in CATEGORIES else "other"
    validated["ai_summary"] = data.get("ai_summary", "") if isinstance(data.get("ai_summary"), str) else ""
    return validated


def extract_article(model, article: dict, max_retries: int = 3) -> dict:
    """Run extraction prompt and return a validated extraction dict.
    Retries up to max_retries times on parse failures or errors.
    """
    raw_title = article.get("title", "")
    if isinstance(raw_title, dict):
        title = raw_title.get("rendered", str(raw_title))
    else:
        title = str(raw_title)

    url = article.get("link", article.get("id", ""))
    published = article.get("published", article.get("_date", ""))
    author = article.get("author", "Unknown")

    raw_content = article.get("content", "")
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)
    content_text = strip_html(raw_content)[:3000]

    prompt = USER_PROMPT_TEMPLATE.format(
        title=title,
        url=url,
        published=published,
        author=author,
        content=content_text,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = model.prompt(prompt, system=SYSTEM_PROMPT)
            raw = response.text()
            parsed = parse_json_response(raw)
            if parsed is not None:
                return validate_extraction(parsed)
            console.print(f"[yellow]Warning: could not parse JSON for '{title[:60]}' (attempt {attempt}/{max_retries})[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Warning: extraction error for '{title[:60]}' (attempt {attempt}/{max_retries}): {exc}[/yellow]")

    return dict(DEFAULT_EXTRACTION)


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict, state_file: Path) -> None:
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f)


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata and summaries from College Park city council articles."
    )
    parser.add_argument(
        "--input",
        default="city_council.json",
        type=Path,
        help="Input JSON file (default: city_council.json).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use for extraction (e.g. groq-llama-3.3-70b).",
    )
    parser.add_argument(
        "--output",
        default="extracted_city_council.json",
        type=Path,
        help="Output JSON file path (default: extracted_city_council.json).",
    )
    parser.add_argument(
        "--state-file",
        default=".extract_city_council_state.json",
        type=Path,
        help="State file for resuming interrupted runs (default: .extract_city_council_state.json).",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Only process the first N articles (useful for testing).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        console.print(f"[red]Error: input file not found: {args.input}[/red]")
        console.print("  Run classify_topics.py first to generate city_council.json.")
        sys.exit(1)

    if args.limit is not None:
        args.output = args.output.with_stem(args.output.stem + "_test")

    console.print("[bold]City Council Extraction[/bold]")
    console.print(f"  Input    : {args.input}")
    console.print(f"  Model    : {args.model}")
    console.print(f"  Output   : {args.output}")
    console.print()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    articles = data.get("articles", [])
    console.print(f"  Loaded {len(articles)} articles.")

    if args.limit is not None:
        articles = articles[: args.limit]
        console.print(f"  [cyan]Limiting to first {len(articles)} articles (--limit).[/cyan]")

    if not articles:
        console.print("[yellow]No articles to process. Exiting.[/yellow]")
        sys.exit(0)

    state = load_state(args.state_file)
    already_done = len(state)
    if already_done:
        console.print(f"  Resuming: {already_done} articles already extracted, skipping.")
    console.print()

    try:
        model = llm.get_model(args.model)
    except Exception as exc:
        console.print(f"[red]Error loading model '{args.model}': {exc}[/red]")
        sys.exit(1)

    failed = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(articles))

        for article in articles:
            article_id = article.get("id") or article.get("link", "")

            # Skip only if previously succeeded (not if it failed)
            if article_id in state and not state[article_id].get("_extraction_failed"):
                progress.advance(task)
                continue

            extraction = extract_article(model, article)
            if extraction.get("_extraction_failed"):
                failed += 1

            state[article_id] = extraction
            save_state(state, args.state_file)
            progress.advance(task)

    # Merge articles with their extractions
    enriched_articles = []
    for article in articles:
        article_id = article.get("id") or article.get("link", "")
        extraction = state.get(article_id, dict(DEFAULT_EXTRACTION))
        enriched = {k: v for k, v in article.items() if k != "content"}
        enriched["extraction"] = extraction
        enriched_articles.append(enriched)

    console.print()
    console.print("[bold green]Extraction complete.[/bold green]")
    console.print(f"  Total articles  : {len(enriched_articles)}")
    if failed:
        console.print(f"  [yellow]Extraction failures: {failed}[/yellow]")

    output_data = {
        "total_articles": len(enriched_articles),
        "model": args.model,
        "articles": enriched_articles,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    console.print(f"\nWrote {len(enriched_articles)} articles to [bold]{args.output}[/bold]")


if __name__ == "__main__":
    main()
