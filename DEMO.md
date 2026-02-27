# Demo: Chicago Local Government News Pipeline

This pipeline classifies and extracts structured metadata from Chicago Sun-Times articles using a large language model. Follow the steps below to run a quick demo in a cloud environment — no local setup required.

---

## 1. Copy the repository

1. Open the [nicar2026 repository](https://github.com/NewsAppsUMD/nicar2026) on GitHub.
2. Click the **"Use this template"** button (top-right of the repo page).
3. Choose **"Create a new repository"**, give it a name, and click **"Create repository"**.

---

## 2. Open a Codespace

1. From your new repository on GitHub, click the green **"Code"** button.
2. Select the **"Codespaces"** tab, then click **"Create codespace on main"**.
3. Wait for the Codespace to finish building and open in your browser.

---

## 3. Install uv and sync dependencies

In the Codespace terminal, install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then reload your shell and install all project dependencies:

```bash
source $HOME/.cargo/env
uv sync
```

---

## 4. Get a Groq API key

1. Go to [https://console.groq.com](https://console.groq.com) and sign up for a free account.
2. Navigate to **API Keys** and create a new key.
3. Copy the key, then set it in the terminal:

```bash
uv run llm keys set groq
```

Paste your key when prompted. You won't see it, but it's there.

---

## 5. Run the classification script

This step fetches articles from GitHub and uses the LLM to identify local government stories. Use `--limit` to process only a small sample:

```bash
cd chicago-public-media
uv run python classify.py --model groq/meta-llama/llama-4-maverick-17b-128e-instruct --limit 10
```

This writes results to `classified_articles_test.json`.

Other model options:
- `groq-kimi-k2`
- `groq/qwen/qwen3-32b`
- `groq/meta-llama/llama-4-scout-17b-16e-instruct`

---

## 6. Run the extraction script

This step reads the classified articles and extracts structured metadata (people, organizations, locations, issues, and a summary):

```bash
uv run python extract.py --model groq/meta-llama/llama-4-maverick-17b-128e-instruct \
    --input classified_articles_test.json \
    --limit 10
```

This writes results to `extracted_articles_test.json`.

---

## Output

Open `extracted_articles_test.json` to see each article enriched with:
- `key_people` — named individuals and their roles
- `organizations` — agencies and institutions mentioned
- `locations` — Chicago neighborhoods, streets, or facilities
- `key_issues` — main policy topics covered
- `category` — one of: `zoning`, `education`, `public_safety`, `budget`, `justice`, `city_council`, `other`
- `ai_summary` — a 2–3 sentence factual summary
