"""Generate a newsroom Beatbook from extracted story summaries.

Reads a JSON file of extracted articles (e.g. extracted_city_council_v2.json),
analyzes trends, people, institutions, and story patterns, then generates a
polished Beatbook in Markdown. 

Usage:
    uv run python generate_beatbook.py \\
        --input extracted_city_council_v2.json \\
        --model groq-llama-3.3-70b \\
        [--output beatbook_college_park.md] \\
        [--json-output beatbook_college_park.json]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import llm
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_articles(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("articles", [])


def get_title(article: dict) -> str:
    t = article.get("title", "")
    if isinstance(t, dict):
        return t.get("rendered", str(t))
    return str(t)


def get_extraction(article: dict) -> dict:
    return article.get("extraction", {})


def parse_date(article: dict) -> datetime | None:
    raw = article.get("published") or article.get("_date", "")
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[:25], fmt[:len(raw[:25])])
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw[:19])
    except Exception:
        return None


def normalize_name(name: str) -> str:
    """Strip title from 'Name, Title' entries and normalize whitespace."""
    return name.split(",")[0].strip().title()


def confidence_label(count: int, total: int) -> str:
    pct = count / total if total else 0
    if pct >= 0.15 or count >= 20:
        return "HIGH"
    if pct >= 0.07 or count >= 10:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(articles: list[dict]) -> dict:
    total = len(articles)
    people_counter: Counter = Counter()
    org_counter: Counter = Counter()
    location_counter: Counter = Counter()
    issue_counter: Counter = Counter()
    category_counter: Counter = Counter()
    people_quotes: dict[str, list[str]] = defaultdict(list)
    org_quotes: dict[str, list[str]] = defaultdict(list)
    issue_quotes: dict[str, list[str]] = defaultdict(list)
    people_titles: dict[str, str] = {}  # name -> most specific title seen

    dated_articles = []
    for article in articles:
        ex = get_extraction(article)
        title = get_title(article)
        summary = ex.get("ai_summary", "")

        for person_entry in ex.get("key_people", []):
            name = normalize_name(person_entry)
            if name and len(name) > 2:
                people_counter[name] += 1
                if summary:
                    people_quotes[name].append(summary)
                # Extract title from "Name, Title" format
                if "," in str(person_entry):
                    title_part = str(person_entry).split(",", 1)[1].strip()
                    # Keep the most descriptive title (longest)
                    if title_part and (name not in people_titles or len(title_part) > len(people_titles[name])):
                        people_titles[name] = title_part

        for org in ex.get("organizations", []):
            if not isinstance(org, str):
                continue
            org = org.strip()
            if org and len(org) > 3:
                org_counter[org] += 1
                if summary:
                    org_quotes[org].append(summary)

        for loc in ex.get("locations", []):
            if not isinstance(loc, str):
                continue
            loc = loc.strip()
            if loc:
                location_counter[loc] += 1

        for issue in ex.get("key_issues", []):
            if not isinstance(issue, str):
                continue
            issue = issue.strip()
            if issue:
                issue_counter[issue] += 1
                if summary:
                    issue_quotes[issue].append(summary)

        cat = ex.get("category", "other")
        if not ex.get("_extraction_failed"):
            category_counter[cat] += 1

        dt = parse_date(article)
        if dt:
            dated_articles.append((dt, article))

    dated_articles.sort(key=lambda x: x[0])

    # Trend detection: split into thirds and compare category frequency
    third = max(1, len(dated_articles) // 3)
    early = dated_articles[:third]
    mid = dated_articles[third:2*third]
    recent = dated_articles[2*third:]

    def period_cats(period):
        c = Counter()
        for _, a in period:
            cat = get_extraction(a).get("category", "other")
            c[cat] += 1
        return c

    early_cats = period_cats(early)
    recent_cats = period_cats(recent)

    rising, fading, persistent = [], [], []
    all_cats = set(early_cats) | set(recent_cats)
    for cat in all_cats:
        e = early_cats.get(cat, 0)
        r = recent_cats.get(cat, 0)
        if r > e * 1.5 and r >= 3:
            rising.append(cat)
        elif e > r * 1.5 and e >= 3:
            fading.append(cat)
        elif e >= 2 and r >= 2:
            persistent.append(cat)

    return {
        "total": total,
        "people": people_counter,
        "people_titles": people_titles,
        "orgs": org_counter,
        "locations": location_counter,
        "issues": issue_counter,
        "categories": category_counter,
        "people_quotes": people_quotes,
        "org_quotes": org_quotes,
        "issue_quotes": issue_quotes,
        "dated_articles": dated_articles,
        "rising": rising,
        "fading": fading,
        "persistent": persistent,
        "early_cats": early_cats,
        "recent_cats": recent_cats,
    }


# ---------------------------------------------------------------------------
# LLM-powered sections
# ---------------------------------------------------------------------------

BEATBOOK_SYSTEM = """You are a narrative journalist and senior editor helping build a
Beatbook for a reporter covering the College Park, Maryland city council beat.

Your writing must tell a story. Every section should read like the opening of a
long-form feature — specific, vivid, and grounded in the facts of the data. Pull
the reader in. Make them feel the texture of this beat: the personalities, the
tensions, the stakes, the unresolved threads.

Style rules:
- Write narratively. Use scene-setting where it helps. Let facts build toward meaning.
- Be specific — names, places, dollar amounts, vote counts, turning points.
- Vary sentence rhythm. Mix short punches with longer, flowing observations.
- Avoid bullet-point thinking even when using lists. Each item should feel alive.
- Never use filler phrases like "it is worth noting", "in conclusion", or "it is important".
- No academic language. No corporate tone. Write like a journalist who loves this beat.
- Preserve every fact accurately. Narrative style must never come at the cost of truth.

Important: This Beatbook is built from archived stories. Some situations, people,
and policy debates may have since moved on — votes taken, officials departed, projects
resolved. Where relevant, flag that reporters should verify current status.

Critical rules — failure to follow these undermines the Beatbook:
- NEVER reference story counts, mention counts, or archive frequency. Do not write
  phrases like "appears 82 times", "dominates the archive", "mentioned frequently",
  or any variation. Convey importance through context and consequence only.
- Do NOT add section headers or titles. The sections already have headers. Start
  your response directly with the prose.
- Cover the full picture — include controversies, failures, criticism, and accountability
  moments alongside achievements. A councilmember who pushed bad policy or faced
  backlash is as important to document as one who passed good legislation.
- Only use roles and titles exactly as they appear in the source data. Do not
  invent, upgrade, or alter anyone's title. If the data says "councilmember",
  do not call them "mayor" or any other role."""


def llm_section(model, prompt: str) -> str:
    try:
        response = model.prompt(prompt, system=BEATBOOK_SYSTEM)
        return response.text().strip()
    except Exception as exc:
        console.print(f"[yellow]Warning: LLM section failed: {exc}[/yellow]")
        return "_[Section generation failed — rerun to retry.]_"


def build_analysis_prompt(section: str, findings: dict, articles: list[dict]) -> str:
    """Build a focused prompt for each beatbook section."""

    top_people = findings["people"].most_common(15)
    top_orgs = findings["orgs"].most_common(10)
    top_issues = findings["issues"].most_common(15)
    top_locations = findings["locations"].most_common(10)
    categories = findings["categories"].most_common()
    total = findings["total"]

    # Gather a sample of summaries for context
    summaries = []
    for article in articles[:60]:
        ex = get_extraction(article)
        s = ex.get("ai_summary", "")
        t = get_title(article)
        if s and t:
            summaries.append(f"- {t}: {s}")
    sample = "\n".join(summaries[:40])

    context = f"""
DATASET: {total} College Park city council stories.

TOP PEOPLE:
{chr(10).join(f"  {name}" for name, _ in top_people)}

TOP ORGANIZATIONS:
{chr(10).join(f"  {org}" for org, _ in top_orgs)}

TOP ISSUES:
{chr(10).join(f"  {issue}" for issue, _ in top_issues)}

TOP LOCATIONS:
{chr(10).join(f"  {loc}" for loc, _ in top_locations)}

TOPIC AREAS: {', '.join(cat for cat, _ in categories)}

RISING TOPICS: {', '.join(findings['rising']) or 'none detected'}
FADING TOPICS: {', '.join(findings['fading']) or 'none detected'}
PERSISTENT TOPICS: {', '.join(findings['persistent']) or 'none detected'}

SAMPLE STORY SUMMARIES:
{sample}
"""

    prompts = {
        "executive_snapshot": f"""
{context}

Write a 3-4 paragraph Executive Snapshot that opens this Beatbook like the first
page of a compelling long-form feature. Draw the reader into the world of College
Park city government — its tensions, its characters, its unresolved fights.

Tell a story. Let the facts build toward meaning. Your writing should cover:
- What this beat is fundamentally about
- The dominant themes and stakes for College Park residents
- Who holds power and why it matters
- What period this data covers and any notable gaps

Reference actual names, issues, and organizations from the data. Make a reporter
who has never covered this beat feel the pulse of it by the time they finish reading.
""",

        "recurring_issues": f"""
{context}

Write a section called "Top Recurring Issues" listing the 5-10 most significant
policy issues on this beat. For each issue:
- Give it a bold header
- Tell the story of why it matters and why it keeps coming up
- Note who the key players are and what's at stake
- Quote or paraphrase a story summary as evidence where it strengthens the writing

Do not reference how many times an issue appeared in the data or use any confidence
ratings. Convey importance through context, consequence, and specificity — not counts.
Write narratively. Each entry should feel alive, not like a database record.
""",

        "power_map": f"""
{context}

Write a "Power Map" section covering the key people and institutions on this beat.

For PEOPLE: Write about the top 8-10 individuals. For each, tell us who they are,
why they matter on this beat, and what issues they're associated with. Write in
full paragraphs, not bullet points. Do not mention how many times they appear in
stories or reference any archive. Convey their significance through their role and
the decisions they've shaped.

For INSTITUTIONS: Cover the top 6-8 organizations. Explain their role in College
Park governance and what they're pushing for or defending.

Format each entry as a bold name followed by a narrative paragraph.
""",

        "geographic_hotspots": f"""
{context}

Write a "Geographic Hotspots" section identifying the specific streets, neighborhoods,
facilities, and zones in College Park that come up most in coverage.

For each location, explain:
- Why it keeps appearing
- What policy issues are tied to it
- Why a reporter should pay attention to it

If the data is thin on locations, say so honestly and suggest where a reporter
should focus their geographic attention based on the issues present.
""",

        "timeline_trends": f"""
{context}

Write a "Timeline and Trend Shifts" section covering how this beat has evolved.

Address what topics are rising and why, what appears to be fading and what that
signals, what has remained persistent, and any notable gaps.

Do not reference the archive, story counts, or data. Write as if you are a
seasoned reporter who has watched this beat unfold over time. Be specific about
what changed and what it means for a reporter working this beat today.
""",

        "undercovered_angles": f"""
{context}

Write an "Undercovered Angles" section identifying gaps in the coverage — stories
that should exist based on what's present in the data but don't seem to have been
fully explored.

Think about:
- People who appear but whose roles are never examined deeply
- Institutions that keep showing up but are never the main focus
- Policy decisions whose downstream effects haven't been tracked
- Communities or neighborhoods that are affected but not quoted
- Budget or financial threads that were mentioned but not followed

Give at least 6 specific undercovered angles with a one-paragraph explanation each.
""",

        "source_development": f"""
{context}

Write a "Source Development Targets" section helping a reporter build their source list.

Identify:
- The 5-6 most important on-the-record sources a reporter must cultivate
- 3-4 institutional sources (offices, agencies) to check regularly
- 2-3 community-level sources (neighborhood groups, advocates, residents)
- Any sources who appear frequently but may need independent verification

For each source, explain why they matter and what access or information they provide.
""",

        "story_ideas": f"""
{context}

Generate exactly 25 high-quality future story ideas for a reporter covering the
College Park city council beat.

For each story idea:
1. Give it a punchy, specific working headline
2. Write 2-3 sentences explaining the story angle
3. Add a "Why now:" line explaining the timeliness
4. Add a "Who to call:" line with specific names or roles from the data

Number each idea 1-25. Make them specific to College Park — not generic local
government story templates. Prioritize ideas that connect multiple threads
(budget + development, public safety + council policy, UMD + city government, etc.).

For each idea, add a brief "Status check:" line flagging whether the reporter
should verify that the situation is still active, ongoing, or unresolved before
pursuing it — since these stories come from an archive and some may have been
overtaken by events.
""",
    }

    return prompts[section]


# ---------------------------------------------------------------------------
# Markdown assembly
# ---------------------------------------------------------------------------

def build_markdown(sections: dict, findings: dict, articles: list[dict]) -> str:
    total = findings["total"]
    dated = findings["dated_articles"]
    date_range = ""
    if dated:
        start = dated[0][0].strftime("%B %Y")
        end = dated[-1][0].strftime("%B %Y")
        date_range = f"{start} – {end}"

    top_cats = findings["categories"].most_common(3)
    cat_summary = ", ".join(f"{c} ({n})" for c, n in top_cats)

    lines = [
        "# College Park City Council Beatbook",
        "",
        f"**Stories analyzed:** {total}  ",
        f"**Date range:** {date_range or 'unknown'}  ",
        f"**Top categories:** {cat_summary}  ",
        f"**Generated:** {datetime.now().strftime('%B %d, %Y')}",
        "",
        "> **Editorial note:** This Beatbook is built from archived stories. Some situations, people, and policy debates described here may have since been resolved, overtaken by events, or changed significantly. Reporters should verify the current status of any story angle before pursuing it.",
        "",
        "---",
        "",
        "## Executive Snapshot",
        "",
        sections.get("executive_snapshot", ""),
        "",
        "---",
        "",
        "## Top Recurring Issues",
        "",
        sections.get("recurring_issues", ""),
        "",
        "---",
        "",
        "## Power Map",
        "",
        sections.get("power_map", ""),
        "",
        "---",
        "",
        "## Geographic Hotspots",
        "",
        sections.get("geographic_hotspots", ""),
        "",
        "---",
        "",
        "## Timeline and Trend Shifts",
        "",
        sections.get("timeline_trends", ""),
        "",
        "---",
        "",
        "## Undercovered Angles",
        "",
        sections.get("undercovered_angles", ""),
        "",
        "---",
        "",
        "## Source Development Targets",
        "",
        sections.get("source_development", ""),
        "",
        "---",
        "",
        "## 25 Future Story Ideas",
        "",
        sections.get("story_ideas", ""),
        "",
        "---",
        "",
        "## Key People",
        "",
    ]

    people_titles = findings.get("people_titles", {})
    for name, _ in findings["people"].most_common(20):
        title = people_titles.get(name, "")
        if title:
            lines.append(f"- **{name}**, {title}")
        else:
            lines.append(f"- **{name}**")

    lines += ["", "## Key Organizations", ""]
    for org, _ in findings["orgs"].most_common(15):
        lines.append(f"- {org}")

    lines += ["", "## Key Locations", ""]
    for loc, _ in findings["locations"].most_common(15):
        lines.append(f"- {loc}")

    lines += [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a College Park City Council Beatbook from extracted articles."
    )
    parser.add_argument(
        "--input",
        default="extracted_city_council_v2.json",
        type=Path,
        help="Input JSON file of extracted articles (default: extracted_city_council_v2.json).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use for generation (e.g. groq-llama-3.3-70b).",
    )
    parser.add_argument(
        "--output",
        default="beatbook_college_park_v5.md",
        type=Path,
        help="Output Markdown file (default: beatbook_college_park_v5.md).",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        type=Path,
        help="Optional JSON companion file with structured findings.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        console.print(f"[red]Error: input file not found: {args.input}[/red]")
        sys.exit(1)

    console.print("[bold]College Park Beatbook Generator[/bold]")
    console.print(f"  Input  : {args.input}")
    console.print(f"  Model  : {args.model}")
    console.print(f"  Output : {args.output}")
    console.print()

    articles = load_articles(args.input)
    console.print(f"  Loaded {len(articles)} articles.")

    if not articles:
        console.print("[red]No articles found. Exiting.[/red]")
        sys.exit(1)

    console.print("  Analyzing...")
    findings = analyze(articles)
    console.print(f"  Found {len(findings['people'])} people, {len(findings['orgs'])} orgs, {len(findings['issues'])} issues.")
    console.print()

    try:
        model = llm.get_model(args.model)
    except Exception as exc:
        console.print(f"[red]Error loading model '{args.model}': {exc}[/red]")
        sys.exit(1)

    section_names = [
        "executive_snapshot",
        "recurring_issues",
        "power_map",
        "geographic_hotspots",
        "timeline_trends",
        "undercovered_angles",
        "source_development",
        "story_ideas",
    ]

    sections = {}
    for name in section_names:
        console.print(f"  Generating: {name.replace('_', ' ').title()}...")
        prompt = build_analysis_prompt(name, findings, articles)
        sections[name] = llm_section(model, prompt)

    console.print()
    console.print("  Assembling Beatbook...")
    markdown = build_markdown(sections, findings, articles)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown)
    console.print(f"[bold green]Beatbook written to {args.output}[/bold green]")

    if args.json_output:
        companion = {
            "generated": datetime.now().isoformat(),
            "total_articles": findings["total"],
            "top_people": findings["people"].most_common(20),
            "top_orgs": findings["orgs"].most_common(15),
            "top_locations": findings["locations"].most_common(15),
            "top_issues": findings["issues"].most_common(15),
            "categories": dict(findings["categories"]),
            "rising_topics": findings["rising"],
            "fading_topics": findings["fading"],
            "persistent_topics": findings["persistent"],
            "sections": sections,
        }
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(companion, f, indent=2, ensure_ascii=False)
        console.print(f"[bold green]JSON companion written to {args.json_output}[/bold green]")


if __name__ == "__main__":
    main()
