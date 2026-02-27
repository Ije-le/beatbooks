"""Stage 2: Extract structured metadata and summaries from classified articles.

Reads classified_articles.json, runs each article through an Ollama model to
extract key people, organizations, locations, issues, a category, and a
2-3 sentence summary. Outputs a combined extracted_articles.json.

Usage:
    uv run python extract.py \\
        --input classified_articles.json \\
        --model llama3.2 \\
        [--output extracted_articles.json] \\
        [--state-file .extract_state.json]
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

CATEGORIES = ["zoning", "education", "public_safety", "budget", "justice", "city_council", "other"]

SYSTEM_PROMPT = """You are a structured data extractor for Chicago local government news.

Given a news article, extract key information and return ONLY a valid JSON object.
Do not include markdown code fences, backticks, or any text outside the JSON object.
Use null for fields where information is not present in the article.
For the category field, choose the single best match from:
zoning, education, public_safety, budget, justice, city_council, other"""

USER_PROMPT_TEMPLATE = """Extract structured metadata from this Chicago local government article.

Title: {title}
URL: {url}
Published: {published}
Author: {author}

Content:
{summary}

Return this exact JSON structure with values filled in:
{{
  "key_people": ["list of names and their roles/titles mentioned"],
  "organizations": ["list of government agencies, departments, or institutions"],
  "locations": ["list of specific Chicago neighborhoods, streets, or facilities"],
  "key_issues": ["list of 2-5 main policy issues or topics covered"],
  "category": "one of: zoning, education, public_safety, budget, justice, city_council, other",
  "ai_summary": "2-3 sentence factual summary of what happened, who is involved, and why it matters for Chicago governance"
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
    """Try to extract a JSON object from the model's response.

    Attempts three strategies:
    1. Direct JSON parse of the stripped response
    2. Extract from markdown code fences (```json ... ```)
    3. Extract first {...} block via brace matching
    """
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


def extract_article(model, article: dict) -> dict:
    """Run extraction prompt and return a validated extraction dict."""
    title = article.get("title", "")
    url = article.get("link", article.get("id", ""))
    published = article.get("published", article.get("_date", ""))
    author = article.get("author", "Unknown")
    raw_summary = article.get("summary", "")
    summary_text = strip_html(raw_summary)[:3000]

    prompt = USER_PROMPT_TEMPLATE.format(
        title=title,
        url=url,
        published=published,
        author=author,
        summary=summary_text,
    )

    try:
        response = model.prompt(prompt, system=SYSTEM_PROMPT)
        raw = response.text()
        parsed = parse_json_response(raw)
        if parsed is None:
            console.print(f"[yellow]Warning: could not parse JSON for '{title[:60]}'[/yellow]")
            return dict(DEFAULT_EXTRACTION)
        return validate_extraction(parsed)
    except Exception as exc:
        console.print(f"[yellow]Warning: extraction error for '{title[:60]}': {exc}[/yellow]")
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
        description="Extract metadata and summaries from classified Chicago articles."
    )
    parser.add_argument(
        "--input",
        default="classified_articles.json",
        type=Path,
        help="Input JSON file from classify.py (default: classified_articles.json).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name to use for extraction (e.g. llama3.2, mistral).",
    )
    parser.add_argument(
        "--output",
        default="extracted_articles.json",
        type=Path,
        help="Output JSON file path (default: extracted_articles.json).",
    )
    parser.add_argument(
        "--state-file",
        default=".extract_state.json",
        type=Path,
        help="State file for resuming interrupted runs (default: .extract_state.json).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        console.print(f"[red]Error: input file not found: {args.input}[/red]")
        console.print("  Run classify.py first to generate the classified articles file.")
        sys.exit(1)

    console.print("[bold]Stage 2: Extraction[/bold]")
    console.print(f"  Input    : {args.input}")
    console.print(f"  Model    : {args.model}")
    console.print(f"  Output   : {args.output}")
    console.print()

    # Load classified articles
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    articles = data.get("articles", [])
    console.print(f"  Loaded {len(articles)} classified articles.")

    if not articles:
        console.print("[yellow]No articles to process. Exiting.[/yellow]")
        sys.exit(0)

    # Load state
    state = load_state(args.state_file)
    already_done = len(state)
    if already_done:
        console.print(f"  Resuming: {already_done} articles already extracted, skipping.")
    console.print()

    # Load model
    try:
        model = llm.get_model(args.model)
    except Exception as exc:
        console.print(f"[red]Error loading model '{args.model}': {exc}[/red]")
        console.print("  Make sure the model is pulled via Ollama and llm-ollama is installed.")
        sys.exit(1)

    # Extract metadata from each article
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

            if article_id in state:
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
        enriched = {k: v for k, v in article.items() if k != "summary"}
        enriched["extraction"] = extraction
        enriched_articles.append(enriched)

    console.print()
    console.print("[bold green]Extraction complete.[/bold green]")
    console.print(f"  Total articles  : {len(enriched_articles)}")
    if failed:
        console.print(f"  [yellow]Extraction failures: {failed}[/yellow]")

    # Write output
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
