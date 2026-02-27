# NICAR 2026: Building Newsroom Products with LLMs

Materials for the 2026 NICAR conference presentation on using Large Language Models to build useful newsroom products from news archives.

The demo in this repo is a **beat book generator** for Chicago local government coverage, built on top of Chicago Sun-Times articles pulled from the [NewsAppsUMD/beat_book_work](https://github.com/NewsAppsUMD/beat_book_work) repository. The pipeline classifies articles, extracts structured metadata, and synthesizes a beat guide — the kind of document that helps a reporter get up to speed on a new beat in minutes instead of months.

---

## The Pipeline

Three Python scripts form the assembly line. Each stage produces a JSON file that feeds the next.

```
classify.py  →  classified_articles.json
extract.py   →  extracted_articles.json
guide.py     →  chicago_local_gov_guide.md
```

### Stage 1: `classify.py` — Filter for relevance

Reads articles from the beat_book_work GitHub archive (or a local directory) and asks a local Ollama model one question per article: is this about Chicago or Cook County local government? Articles that don't make the cut are dropped.

Supports resuming interrupted runs via a state file.

```bash
# Pull articles from GitHub and classify
uv run python chicago-public-media/classify.py --model llama3.2

# Or use a local checkout
uv run python chicago-public-media/classify.py \
    --data-dir /path/to/beat_book_work/chicago-public-media/data/2026 \
    --model llama3.2
```

### Stage 2: `extract.py` — Pull out structure

Runs each classified article through Ollama to extract:

- Key people (with roles)
- Organizations and agencies
- Chicago neighborhoods and locations
- Policy issues
- A category (`zoning`, `education`, `public_safety`, `budget`, `justice`, `city_council`, or `other`)
- A 2–3 sentence factual summary

```bash
uv run python chicago-public-media/extract.py \
    --input chicago-public-media/classified_articles.json \
    --model llama3.2
```

### Stage 3: `guide.py` — Synthesize the beat book

Sends the full corpus of extracted articles to Claude and asks it to act as a veteran Chicago investigative editor writing a beat guide for a reporter new to the city. The output is a narrative Markdown document with inline citations linking back to the original articles.

The guide covers:
- The political landscape (mayor, council factions, fault lines)
- Major ongoing policy battles
- Key people to know
- Institutions and how power flows through them
- Neighborhoods and communities on the agenda
- Story ideas and threads worth pursuing

```bash
uv run python chicago-public-media/guide.py \
    --input chicago-public-media/extracted_articles.json
```

Requires an Anthropic API key:

```bash
llm keys set anthropic
# or set ANTHROPIC_API_KEY in your environment
```

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dwillis/nicar2026.git
cd nicar2026
uv sync
source .venv/bin/activate
```

[Ollama](https://ollama.com) must be running locally for stages 1 and 2:

```bash
ollama pull llama3.2
```

---

## The Source Data

Articles come from [NewsAppsUMD/beat_book_work](https://github.com/NewsAppsUMD/beat_book_work), a repository of Chicago Sun-Times RSS feed data updated daily by a GitHub Actions workflow. Each day's articles live at `chicago-public-media/data/2026/YYYY-MM-DD/YYYY-MM-DD.json`.

`classify.py` fetches this data directly from GitHub by default, so no local clone of that repo is required.

---

## Key Design Choices

**Two models, two jobs.** Stages 1 and 2 use small local Ollama models (fast, cheap, no API key needed). Stage 3 uses Claude, which is better at long-context synthesis and prose generation.

**State files for resilience.** Stages 1 and 2 write a `.json` state file after each article. If a run is interrupted, it resumes exactly where it left off — important when classifying hundreds of articles.

**No `summary` in the final output.** Raw article HTML is used during classification and extraction but stripped from `extracted_articles.json` to keep the file readable and manageable.

**Prompt as editorial voice.** The `guide.py` system prompt is as much editorial policy as LLM instruction — it explicitly bans AI-editorial phrases ("underscores the importance of," "navigating the complexities") and tells the model to write like a reporter, not a commentator.

---

## Output Example

A sample beat guide generated from articles published in February 2026 is in [`chicago-public-media/chicago_local_gov_guide.md`](chicago-public-media/chicago_local_gov_guide.md).
