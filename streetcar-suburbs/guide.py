"""Stage 3: Synthesize a Chicago local government beat guide using Claude.

Reads extracted_articles.json, builds a structured brief for each article,
then asks Claude to synthesize a comprehensive beat guide organized by theme
with inline citations linking to the original articles.

Usage:
    uv run python guide.py \\
        --input extracted_articles.json \\
        [--model claude-sonnet-4.6] \\
        [--output chicago_local_gov_guide.md]

Requires ANTHROPIC_API_KEY environment variable or:
    uv run llm keys set anthropic
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import llm
from rich.console import Console

console = Console()

SYSTEM_PROMPT = """You are a veteran Chicago investigative editor with 20 years covering City Hall, the
Cook County courthouse, and Chicago's neighborhoods. You are writing a beat guide to help a
journalist new to Chicago local government quickly get up to speed.

Your goal is NOT to catalog or summarize every article. Your goal is to SYNTHESIZE — to find
the patterns, identify the fault lines, name the power players, and point reporters toward
the stories that matter most. Draw connections between issues. Highlight
contradictions and tensions. Tell the reporter what they need to watch.

Write in clear, direct prose. Use inline citations in the format ([Headline](URL)) when
referencing specific reporting. Every factual claim about an event should be anchored to a
specific article citation.

<writing_style>
Maintain prose that is direct, clear, and substantive:

- **Short paragraphs**: 2-4 sentences max. One idea per paragraph. White space helps readers.
- **Natural flow over formality**: Write like you're explaining something to a smart colleague. Vary sentence length.
- **Precision without stiffness**: Concrete details, specific names, but avoid mechanical institutional cadence.
- **No filler**: Cut throat-clearing phrases. Start with the information.
- **Active voice, strong verbs**: Direct constructions over passive.

**CRITICAL - Remove AI-generated editorializing:**
If you see phrases like these, DELETE them or rewrite to just state facts:
- "at the intersection of," "serves as a reminder," "underscores the importance," "highlights the challenges," "reflects broader trends," "speaks to," "emblematic of," "a microcosm of," "navigating the complexities"
- Anything telling the reader what something "means" or "suggests"
- Framing things as "notable," "significant," or "important"

You are a reporter, not a commentator. State what happened. Let readers draw their own conclusions.
</writing_style>
"""

GUIDE_PROMPT_TEMPLATE = """Below is a collection of Chicago Sun-Times articles about local government published in early 2026.
Each article includes an AI-generated summary, key people, organizations, locations, issues, and a URL.

Using this corpus, write a comprehensive beat guide for a reporter new to covering Chicago local government.
Structure your guide with these six sections:

## The Political Landscape
Map the key power centers: Who is Mayor Brandon Johnson and what are his priorities? Who are
the most influential aldermen and what factions exist in the City Council? What are the major
fault lines — between the mayor and the council, between City Hall and Springfield, between
neighborhoods and downtown? Ground every claim in the articles below.

## Major Issues and Ongoing Battles
Cover 4-6 of the most significant policy issues appearing in the corpus. For each issue:
- What's at stake for Chicago residents
- Which officials or factions are on each side
- What has happened recently (cite specific articles)
- What questions remain unanswered or unresolved

## Key People to Know
Profile 5-6 recurring figures from the corpus: their official role, their political position,
their agenda, and why a reporter needs to track them. Be specific — not just "an alderman"
but "the 47th Ward alderman who is blocking the mayor's zoning package."

## Follow the Money
Identify the financial angles in this coverage: budget decisions, TIF district fights, contract
awards, tax policy, development deals, fines and settlements. Where is money moving and who
controls it?

## Stories Worth Chasing
Based on gaps, tensions, and threads in this corpus, identify 4-6 stories that aren't fully
told yet. What questions does this coverage raise but not answer? What patterns suggest a
bigger story? What watchdog angles are hiding in plain sight?

## Quick Reference: Who Does What
List the key government bodies and departments that appear in this coverage with a one-sentence
explanation of their role and why a reporter should know them.

---

ARTICLE CORPUS:

{article_blocks}"""


def build_article_block(article: dict) -> str:
    """Format a single article as a structured text block for the prompt."""
    title = article.get("title", "Untitled")
    url = article.get("link", article.get("id", ""))
    date = article.get("_date", article.get("published", "Unknown date"))
    author = article.get("author", "Unknown")
    extraction = article.get("extraction", {})

    ai_summary = extraction.get("ai_summary", "")
    key_people = extraction.get("key_people", [])
    organizations = extraction.get("organizations", [])
    locations = extraction.get("locations", [])
    key_issues = extraction.get("key_issues", [])
    category = extraction.get("category", "other")

    lines = [
        f"### {title}",
        f"URL: {url}",
        f"Date: {date} | Author: {author} | Category: {category}",
    ]
    if ai_summary:
        lines.append(f"Summary: {ai_summary}")
    if key_people:
        lines.append(f"Key people: {'; '.join(key_people)}")
    if organizations:
        lines.append(f"Organizations: {'; '.join(organizations)}")
    if locations:
        lines.append(f"Locations: {'; '.join(locations)}")
    if key_issues:
        lines.append(f"Issues: {'; '.join(key_issues)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Chicago local government beat guide using Claude."
    )
    parser.add_argument(
        "--input",
        default="extracted_articles.json",
        type=Path,
        help="Input JSON file from extract.py (default: extracted_articles.json).",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4.6",
        help="Claude model to use (default: claude-sonnet-4.6).",
    )
    parser.add_argument(
        "--output",
        default="chicago_local_gov_guide.md",
        type=Path,
        help="Output markdown file (default: chicago_local_gov_guide.md).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        console.print(f"[red]Error: input file not found: {args.input}[/red]")
        console.print("  Run extract.py first to generate the extracted articles file.")
        sys.exit(1)

    console.print("[bold]Stage 3: Guide Generation[/bold]")
    console.print(f"  Input    : {args.input}")
    console.print(f"  Model    : {args.model}")
    console.print(f"  Output   : {args.output}")
    console.print()

    # Load extracted articles
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    articles = data.get("articles", [])
    ollama_model = data.get("model", "unknown")

    console.print(f"  Loaded {len(articles)} extracted articles (classified by: {ollama_model}).")

    if not articles:
        console.print("[yellow]No articles to process. Exiting.[/yellow]")
        sys.exit(0)

    # Filter out articles where extraction failed and have no summary
    usable = [
        a for a in articles
        if not a.get("extraction", {}).get("_extraction_failed")
        or a.get("extraction", {}).get("ai_summary")
    ]
    skipped = len(articles) - len(usable)
    if skipped:
        console.print(f"  [yellow]Skipping {skipped} articles with failed extractions.[/yellow]")
    console.print(f"  Using {len(usable)} articles for synthesis.")
    console.print()

    # Build article blocks
    article_blocks = "\n\n---\n\n".join(build_article_block(a) for a in usable)

    # Build the full prompt
    guide_prompt = GUIDE_PROMPT_TEMPLATE.format(article_blocks=article_blocks)

    # Load model and generate guide
    try:
        model = llm.get_model(args.model)
    except Exception as exc:
        console.print(f"[red]Error loading model '{args.model}': {exc}[/red]")
        console.print(
            "  Make sure llm-anthropic is installed and ANTHROPIC_API_KEY is set."
        )
        console.print("  You can set it with: uv run llm keys set anthropic")
        sys.exit(1)

    console.print(f"Calling [bold]{args.model}[/bold] to synthesize guide...")
    console.print("(This may take a minute for a large corpus.)")
    console.print()

    try:
        response = model.prompt(guide_prompt, system=SYSTEM_PROMPT, max_tokens=16000)
        guide_body = response.text()
    except Exception as exc:
        console.print(f"[red]Error generating guide: {exc}[/red]")
        sys.exit(1)

    # Build the final markdown document
    date_range_start = articles[0].get("_date", "") if articles else ""
    date_range_end = articles[-1].get("_date", "") if articles else ""
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    header = f"""# Chicago Local Government Beat Guide

**Coverage period:** {date_range_start} – {date_range_end}
**Articles analyzed:** {len(usable)} local government stories
**Extraction model:** {ollama_model}
**Synthesis model:** {args.model}
**Generated:** {generated_at}

> This guide was synthesized by AI from Chicago Sun-Times articles. All factual claims
> are grounded in the cited articles. Verify independently before publishing.

---

"""

    full_guide = header + guide_body

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(full_guide)

    console.print(f"[bold green]Guide complete.[/bold green]")
    console.print(f"Wrote beat guide to [bold]{args.output}[/bold]")


if __name__ == "__main__":
    main()
