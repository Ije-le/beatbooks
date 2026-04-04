# How We Built the College Park City Council Beatbook

This document walks through exactly what we did to go from a raw dataset of news articles to a finished Beatbook. If you're picking this up for the first time, start here.

---

## The Big Picture

We started with roughly 2,000 news articles from the Chicago Sun-Times covering streetcar suburb communities across the Chicago area. The goal was to build a Beatbook focused specifically on College Park, Maryland — so the first challenge was finding the College Park stories buried in that larger dataset. From there, we extracted structured information from those stories and used it to generate the Beatbook.

The whole process runs in three stages: **classify**, **extract**, and **generate**. Each stage feeds into the next.

---

## Where the Data Lives

All the articles live in a single JSON file hosted on GitHub:

```
NewsAppsUMD/beat_book_work → streetcar-suburbs/streetcarsuburbs.json
```

The scripts fetch this file automatically when you run them — you don't need to download it manually. If you'd rather work from a local copy, every script accepts a `--data-file` flag pointing to a local path.

---

## Stage 1: Classification

### What we were trying to do

Before we could build anything useful, we needed to separate the College Park stories from everything else. The dataset covers many suburbs — West Chicago, Hyattsville, and others — so most articles aren't relevant to our beat.

We wrote two classification scripts. The first asks a simple yes/no question about each article. The second goes further and assigns topic labels.

### `classify_streetcar.py` — Is this a College Park story?

This script reads every article's title and full content, then asks an AI model one question: does this story involve the College Park, Maryland city government? That means the city council, the mayor, council members, city votes, ordinances, city budget, city services — anything that's specifically College Park city governance, not the county, not the state, not a neighboring suburb.

The script saves its progress after every single article into a state file (`.classify_streetcar_state.json`). That means if it gets interrupted — a rate limit, a crash, you pressing Ctrl+C — you can just rerun it and it picks up exactly where it stopped. Only articles that haven't been classified yet get sent to the model.

One thing to know: if you update the classification prompt (say, to broaden or narrow what counts as a College Park story), you need to delete the state file first. Otherwise the script will use the old cached results for everything it already processed.

```
TERM=dumb uv run classify_streetcar.py --model groq-llama-3.3-70b
```

Output: `classified_streetcar.json`

### `classify_topics.py` — What is each story about?

Once we knew which stories touched College Park, we wanted to understand what kinds of stories they were. This script classifies every article in the full dataset into one or more topic buckets — but only saves it as a meaningful topic if it's specifically about College Park. Anything that isn't gets filed as `other`.

The topics we defined are: **city_council, public_safety, education, arts_culture, food, business, security, sports, other.**

A story can belong to more than one topic — a city council vote on a school closure, for example, would land in both `city_council` and `education`. At the end of the run, the script writes one JSON file per topic. So you end up with `city_council.json`, `education.json`, `arts_culture.json`, and so on.

The same state file logic applies here — progress is saved after every article, and the script resumes cleanly after any interruption.

```
TERM=dumb uv run classify_topics.py --model groq-llama-3.3-70b
```

Output: `city_council.json`, `education.json`, `arts_culture.json`, etc.

---

## Stage 2: Extraction

### What we were trying to do

Classification told us *which* stories were relevant. Extraction tells us *what's in them*. For the Beatbook, we needed more than just the article text — we needed to know who was mentioned, what institutions kept showing up, where events were happening, and what the core policy issues were.

### `extract_city_council.py` — Pull the key details out of each story

This script reads `city_council.json` and sends each article through an AI model with a specific set of instructions: pull out the key people (with their full titles and organizational affiliations), the institutions involved, the specific locations mentioned, the main policy issues, a category label, and a 2–3 sentence summary.

A few things we were deliberate about:
- **People must have full titles.** Not just "Mayor" — "College Park Mayor." Not just "Councilmember" — "College Park City Councilmember." The goal was to be specific enough that a reporter reading the Beatbook knows exactly who held what role.
- **No author names in the people list.** The person who wrote the article is not a subject of it.
- **No publication names in the organizations list.** We want institutions, not media outlets.
- **The article content is used as input but not passed through to the output.** The full text was only there to inform the extraction — the output stays clean.

If the model returns a response it can't parse as valid JSON (which happens occasionally), it tries up to three times before giving up. On the next run, it automatically retries any articles that previously failed, so you don't have to delete the state file to fix those gaps.

```
TERM=dumb uv run extract_city_council.py --model groq-llama-3.3-70b
```

Output: `extracted_city_council_v2.json` — a flat JSON array, one object per article, with no nesting. This format was intentional: it makes the file easy to import directly into datasette or a SQLite database without any reformatting.

---

## Stage 3: Beatbook Generation

### What we were trying to do

With structured data in hand, we had everything we needed to write the Beatbook. The generation script does two things: first, it analyzes the data locally to find patterns (who keeps showing up, what issues recur, how topics have shifted over time); then it hands that analysis to an AI model and asks it to write each section of the Beatbook.

### `generate_beatbook.py` — Write the Beatbook

The script reads `extracted_city_council_v2.json` and starts by doing its own analysis without any AI involvement — counting people, organizations, locations, and issues; detecting which topics are rising, fading, or persistent by comparing earlier stories to more recent ones; and building a map of each person's title from the extracted metadata.

All of that analysis gets packaged into a detailed prompt for each Beatbook section. The AI then writes the prose. We generate eight sections this way, plus a locally-built appendix of key people, organizations, and locations.

The writing style we specified is narrative and journalistic — not a data summary, not a listicle. Each section should read like it was written by a reporter who knows this beat deeply, not like a machine cataloguing a database. Story counts and confidence ratings don't appear anywhere in the prose. Importance is conveyed through context and consequence.

One editorial note baked into every section: since this Beatbook is built from archived stories, some situations described may have since been resolved or overtaken by events. The script instructs the model to flag story angles that reporters should verify before pursuing.

```
TERM=dumb uv run generate_beatbook.py --model claude-sonnet-4-5
```

Output: `beatbook_college_park_v4.md`

To also save a structured JSON companion file:
```
TERM=dumb uv run generate_beatbook.py --model claude-sonnet-4-5 --json-output beatbook_college_park.json
```

The Beatbook sections are:
1. Executive Snapshot
2. Top Recurring Issues
3. Power Map (key people and institutions)
4. Geographic Hotspots
5. Timeline and Trend Shifts
6. Undercovered Angles
7. Source Development Targets
8. 25 Future Story Ideas — each with a working headline, the angle, a "Why now" note, a "Who to call" line, and a "Status check" flag

---

## Viewing the Data in Datasette

If you want to browse the extracted articles interactively — filter by category, search for a person's name, look at all stories from a given location — you can load the data into datasette.

First, install the tools if you haven't:
```
pip install datasette sqlite-utils
```

Then import the articles into a SQLite database:
```
python3 -c "import json; data=json.load(open('streetcar-suburbs/extracted_city_council_v2.json')); import sqlite_utils; db=sqlite_utils.Database('city-council.db'); db['stories'].insert_all(data)"
```

Then launch datasette:
```
datasette city-council.db
```

Open `http://localhost:8001` in your browser. Make sure it's `http://` not `https://` — the browser may try to force a secure connection and fail.

---

## Quick Reference

### Scripts

| Script | Reads from | Writes to | What it does |
|---|---|---|---|
| `classify_streetcar.py` | GitHub JSON | `classified_streetcar.json` | Filters for College Park city government stories |
| `classify_topics.py` | GitHub JSON | `city_council.json` etc. | Sorts all articles into topic buckets |
| `extract_city_council.py` | `city_council.json` | `extracted_city_council_v2.json` | Extracts people, orgs, locations, issues, summaries |
| `generate_beatbook.py` | `extracted_city_council_v2.json` | `beatbook_college_park_v4.md` | Generates the finished Beatbook |

### State Files

Each script saves its progress here. If a run is interrupted, just rerun the same command — it will resume. Delete the state file only if you want to start completely fresh.

| Script | State file |
|---|---|
| `classify_streetcar.py` | `.classify_streetcar_state.json` |
| `classify_topics.py` | `.classify_topics_state.json` |
| `extract_city_council.py` | `.extract_city_council_v2_state.json` |
