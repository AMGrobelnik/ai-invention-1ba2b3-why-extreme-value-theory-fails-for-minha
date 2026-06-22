# gen_art_dataset_1 — test_idea

> Phase: `invention_loop` · round 1 · `gen_art`
> Run: `run_EqcgJR2naF4b` — Why Extreme Value Theory Fails for MinHash Confidence Intervals on Short Text (and What to Do Instead)
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_art_dataset_1` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 05:08:29 UTC

```
<workspace>
Your workspace: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/`:
GOOD: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/file.py`, `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>
<user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>
<artifact_plan>
id: gen_plan_dataset_1_idx2
type: dataset
title: Short Text Datasets for EVT-MinHash Near-Duplicate Detection Evaluation
summary: >-
  Collect and standardize 3+ short text datasets (tweets, SMS, news headlines) with 10K+ documents each, containing 10-100
  words per document, for evaluating MinHash-based near-duplicate detection with EVT-based confidence intervals.
runpod_compute_profile: gpu
ideal_dataset_criteria: >-
  Datasets must contain short text documents (10-100 words each) suitable for near-duplicate detection evaluation. Ideal datasets:
  (1) Twitter/tweet datasets with 10K+ tweets of 10-50 words, (2) SMS/messaging datasets with 10K+ messages of 5-30 words,
  (3) News headlines datasets with 10K+ headlines of 5-20 words. Documents should be real user-generated content (not synthetic).
  Text should be in English or easily translatable. Dataset size must be <300MB when downloaded. Must have clear text field
  and ability to assign unique doc_id. Shingle count (k=3 character shingles) should range from 8-95 for 10-100 word documents.
dataset_search_plan: "PHASE 1: Search HuggingFace Hub for candidate datasets\n\n1. Search for tweet datasets:\n   - Query:\
  \ 'tweet' → filter for datasets with 'text' field and short documents\n   - Specific candidates: 'tweet_eval', 'cardiffnlp/tweet_sentiment_multilingual',\
  \ 'twitter_sentiment'\n   - Preview each to verify: (a) text field exists, (b) average word count 10-100, (c) 10K+ rows\n\
  \n2. Search for SMS/messaging datasets:\n   - Query: 'sms' or 'text messaging' → look for SMS spam or messaging corpora\n\
  \   - Specific candidates: 'sms_spam_collection', 'uci/sms_spam', search for 'NUS SMS Corpus'\n   - Preview to verify text\
  \ length and volume\n\n3. Search for news headlines datasets:\n   - Query: 'news headlines' or 'ag_news' or 'news titles'\n\
  \   - Specific candidates: 'ag_news' (use only titles), 'bbc_news_summary', 'news_category_dataset'\n   - Preview to verify\
  \ titles/headlines are 5-20 words\n\nPHASE 2: Web search for additional datasets (if HuggingFace insufficient)\n\n1. Search\
  \ for 'Tweet dataset CSV download' or 'SMS dataset UCI'\n2. Check UCI ML repository for SMS Spam Collection\n3. Check Kaggle\
  \ for 'Twitter sentiment' or 'SMS spam' datasets\n4. Search for 'news headlines dataset' from sources like BBC, Reuters\n\
  \nPHASE 3: Download and preview promising candidates\n\n1. For each candidate dataset:\n   a. Run aii_hf_preview_datasets.py\
  \ to inspect structure\n   b. Check: number of rows, text field name, sample text lengths\n   c. Calculate word count distribution\
  \ on sample\n   d. Verify <300MB total size\n\n2. Select top 3-5 datasets meeting criteria:\n   - 10K+ documents\n   - 10-100\
  \ words per document (80th percentile)\n   - Clear text field\n   - Loadable via HuggingFace or direct download\n\nPHASE\
  \ 4: Download full datasets and standardize\n\n1. Download each selected dataset using aii_hf_download_datasets.py\n2. Standardize\
  \ to JSON format with REQUIRED fields:\n   {\n     \"doc_id\": \"unique_string\",\n     \"text\": \"document text content\"\
  ,\n     \"source\": \"dataset_name\",\n     \"length\": {\n       \"words\": 45,\n       \"shingles_k3\": 128\n     }\n\
  \   }\n\n3. For each dataset, compute:\n   a. doc_id: Use dataset's ID field or generate sequential IDs\n   b. text: Extract\
  \ and clean text field (remove HTML, normalize whitespace)\n   c. source: Dataset name (e.g., 'tweet_eval', 'sms_spam')\n\
  \   d. length.words: Count words (split on whitespace, filter empty)\n   e. length.shingles_k3: Count unique 3-character\
  \ shingles\n\n4. Filter documents:\n   - Keep only documents with 10-100 words\n   - If >50% of documents outside range,\
  \ try different preprocessing or reject dataset\n\nPHASE 5: Create dataset splits\n\nFor each dataset, create three files:\n\
  1. preview_{dataset}.json: 3 rows with truncated text (100 chars max)\n2. mini_{dataset}.json: 100 random rows (full text)\n\
  3. full_{dataset}.json: All rows (full text)\n\nPHASE 6: Validation and fallback\n\n1. Validate each dataset:\n   - Check\
  \ total document count (warn if <5K, error if <1K)\n   - Verify JSON schema matches required fields\n   - Check word count\
  \ distribution (histogram)\n   - Sample 10 documents and manually verify quality\n\n2. If insufficient datasets found:\n\
  \   FALLBACK OPTION A: Generate synthetic near-duplicates\n   - Take 1K base documents from any dataset\n   - Create near-duplicates\
  \ by: (a) deleting 10-30% of words, (b) replacing synonyms, (c) reordering sentences\n   - This creates pairs with known\
  \ Jaccard similarity\n   \n   FALLBACK OPTION B: Sample from larger datasets\n   - Use 'bookcorpus' or 'wikipedia' but extract\
  \ short paragraphs/sentences\n   - Filter for length 10-100 words\n\n   FALLBACK OPTION C: Combine multiple small datasets\n\
  \   - Merge 3-4 smaller datasets (each 2-5K docs) to reach 10K+\n\n3. Final deliverable: 3+ datasets in data_out.json format\n\
  \   - Merge all datasets into single data_out.json with field 'dataset_name' to identify source\n   - Include metadata:\
  \ total_docs, avg_words, source_datasets\n\nEXECUTION NOTES:\n- Use aii-hf-datasets skill for HuggingFace datasets (search/preview/download)\n\
  - Use aii-web-tools skill for web searches if HuggingFace insufficient\n- Use aii-json skill to validate output JSON schema\n\
  - Use aii-file-size-limit skill to check output files don't exceed limits\n- All processing in Python 3.12 with pandas/numpy\
  \ for data manipulation\n- Maximum 300MB total dataset size (enforced by preview step)\n- Budget: $0 LLM API calls (no OpenRouter\
  \ needed for dataset collection)\n\nSPECIFIC DATASET RECOMMENDATIONS (start here):\n\n1. tweet_eval (HuggingFace: tweet_eval)\n\
  \   - Contains tweets for emotion/sentiment tasks\n   - ~10K tweets per subset, ~10-30 words each\n   - Preview: aii_hf_preview_datasets.py\
  \ tweet_eval --config emotion --num-rows 10\n\n2. sms_spam_collection (HuggingFace: uci/sms_spam or sms_spam_collection)\n\
  \   - SMS messages, ~5-30 words each\n   - ~5.5K messages (may need to combine with another dataset)\n   - Preview: aii_hf_preview_datasets.py\
  \ uci/sms_spam --num-rows 10\n\n3. ag_news (HuggingFace: ag_news)\n   - News articles with titles (use only titles field)\n\
  \   - ~120K articles, titles are 5-15 words\n   - Preview: aii_hf_preview_datasets.py ag_news --num-rows 10\n\n4. bbc_news_summary\
  \ (HuggingFace: bbc_news_summary)\n   - BBC news headlines and summaries\n   - Use headlines (10-20 words)\n   - Preview:\
  \ aii_hf_preview_datasets.py bbc_news_summary --num-rows 10\n\n5. cardiffnlp/tweet_sentiment_multilingual (HuggingFace)\n\
  \   - Multilingual tweets, filter for English\n   - ~12K tweets, 10-40 words\n   - Preview: aii_hf_preview_datasets.py cardiffnlp/tweet_sentiment_multilingual\
  \ --num-rows 10"
target_num_datasets: 3
</artifact_plan>



<available_resources>
<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>

<skills>
Skills are self-contained capabilities with instructions, context, and tools.

- aii-web-tools: Web search (Serper), page/PDF fetch as markdown, regex grep over page/PDF text
- aii-semscholar-bib: Batch-fetch BibTeX from Semantic Scholar
- aii-openrouter-llms: Search and call 300+ LLMs via OpenRouter
- aii-hf-datasets: Search, preview, download HuggingFace datasets
- aii-owid-datasets: Search and load Our World in Data tables
- aii-lean: Compile/verify Lean 4 code, Mathlib search, tactic suggestions
- aii-image-gen: Generate/edit images via Gemini 3 Pro Image (Nano Banana Pro)
- aii-json: Validate JSON against schemas, generate mini/preview variants
- aii-paper-writing: Academic paper structure, bibliography, citations
- aii-paper-to-latex: Assemble LaTeX papers and compile to PDF
- aii-parallel-computing: GPU acceleration, CPU parallelism, async I/O
- aii-python: Python coding standards for experiment scripts
- aii-use-hardware: Detect CPU/RAM/GPU, memory-safe processing
- aii-long-running-tasks: Gradual scaling pattern for long-running tasks
- aii-colab: Google Colab runtime constraints for notebooks
- aii-file-size-limit: Check and split oversized output files
- aii-handbook-multi-llm-agents: Multi-LLM agent orchestration patterns
</skills>
</available_resources>

<available_data_sources>
Use the sources appropriate to your task. Read the relevant skill file BEFORE using each source.

- **HuggingFace Hub** (HF) — ML datasets (NLP, vision, tabular, benchmarks)
- **Our World in Data** (OWID) — Global statistics (energy, health, economics, environment, demographics)
- **Alternate methods** — Python/shell (sklearn.datasets, openml, direct URL, APIs, etc.)

If the plan specifies a source or one fits better, use it.
You may combine sources. Use web search (aii-web-tools skill) to research candidates (background, papers, provenance) — NOT to find/download datasets.
</available_data_sources>

<available_domain_handbooks>
If your domain has a handbook, read the relevant skill file BEFORE working on that domain.

- **Multi-LLM Agents** — dataset selection, evaluation metrics, agent orchestration patterns
</available_domain_handbooks>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<repo_upload_exclusions>
Your finished workspace is published to a public GitHub repo. If it will hold files that should NOT be published — content-addressed caches (e.g. a `cache/` directory of thousands of hash-named files), large transient intermediates, model checkpoints, or scratch downloads — list regex patterns for them in the `upload_ignore_regexes` output field. Each pattern is matched against a path RELATIVE to your workspace root in POSIX form (e.g. `(^|/)cache/`, `(^|/)checkpoints/`). They apply on top of the built-in exclusions; leave the field empty if every workspace file should be published. Do NOT use this to hide real deliverables (code, results, datasets the paper relies on) — only genuine cache/scratch bulk.
</repo_upload_exclusions>

IMPORTANT: Your final response should be at most 300 characters long.

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Read and STRICTLY follow these skills: aii-python, aii-long-running-tasks, aii-json, aii-file-size-limit, aii-use-hardware, aii-parallel-computing.
TODO 2. Read skill files for your data sources (see <available_data_sources>) and domain handbook if applicable (see <available_domain_handbooks>). Based on plan and context, decide which source(s) to use. Include everything specified in the artifact plan, but you may also collect additional relevant data beyond what's listed. Run 24 diverse searches across chosen source(s) — BROAD, GENERAL terms, not very specific. Parallelize where supported.
TODO 3. Identify the 12 most promising datasets. IMPORTANT: Only consider datasets under 300MB. Preview/inspect sample rows for each candidate. Parallelize previews.
TODO 4. Research each candidate BEFORE choosing which to download. For each, search the web (aii-web-tools skill): dataset name, papers citing it, original source/task, popularity. Red flags: no search results, no papers, anonymized features (F1, F2...), <100 downloads, no documentation. Green flags: papers using it, clear documentation, meaningful features, established benchmark. Also consider: will features/structure allow meaningful evaluation of the planned method?
TODO 5. Decide which to KEEP vs DISCARD. Look for: clear structure, relevant fields, quality examples matching requirements, confirmed provenance. Determine which 6 datasets have the most suitable data. Download and save to `temp/datasets/`. Parallelize downloads.
</todos>
```

### [2] HUMAN-USER prompt · 2026-06-22 05:08:29 UTC

```
Build and evaluate a simple MinHash near-duplicate detector for short text documents.
```

### [3] SKILL-INPUT — aii-hf-datasets · 2026-06-22 05:08:41 UTC

The agent loaded the **aii-hf-datasets** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-hf-datasets
description: Searches, previews, and downloads datasets from HuggingFace Hub. Use when user needs machine learning datasets, training data, HuggingFace datasets, dataset discovery, or .parquet/.json exports.
---

## Contents

- Workflow (3-phase dataset discovery)
- Scripts (Search, Preview, Download)

**IMPORTANT - Parallel execution:** GNU `parallel` subshells do NOT inherit `source activate`. Use `export` for variables and **single-quoted** command templates so parallel's subshells can resolve them:
```
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-hf-datasets"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
```

---

## Workflow: 3-Phase Dataset Discovery

### Phase 1: Search for Datasets
Find datasets with metadata (configs, splits, features, sizes)
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-hf-datasets" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_hf_search_datasets.py --query "sentiment analysis" --limit 5
```

### Phase 2: Preview Dataset (if promising)
Inspect metadata AND sample rows in one call
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-hf-datasets" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_hf_preview_datasets.py openai/gsm8k
```

### Phase 3: Download Dataset (if suitable)
Download after reviewing the preview
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-hf-datasets" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_hf_download_datasets.py openai/gsm8k --config main --split train
```

---

## Scripts

### Search HuggingFace Datasets (aii_hf_search_datasets.py)

Search and discover datasets on HuggingFace Hub.

**Example input:**
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-hf-datasets" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_hf_search_datasets.py --query "text classification" --limit 5
```

**Parallel execution (multiple queries):**

IMPORTANT: Use full python path with GNU parallel (venv activate does NOT work in parallel subshells):
```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-hf-datasets" && \
export PY="$SKILL_DIR/../.ability_client_venv/bin/python" && \
export S="$SKILL_DIR/scripts/aii_hf_search_datasets.py" && \
parallel -j 10 -k --group --will-cite '$PY $S --query {} --limit 3' ::: 'sentiment' 'classification' 'translation'
```

**Example output:**
```
Found 5 dataset(s) for query='text classification'

============================================================
Dataset 1: stanfordnlp/imdb
Downloads: 2,500,000 | Likes: 1,234
Description: Large Movie Review Dataset for binary sentiment classification...
Tags: text-classification, en, sentiment-analysis
```

**Result fields per dataset:**

Each entry in ``results`` carries:

- ``id`` / ``downloads`` / ``likes`` / ``tags`` / ``description`` — standard
  HF metadata
- ``has_loader_script`` (bool) — repo ships a top-level ``<repo>.py`` loader.
  ``datasets>=3`` won't run these directly; the dataset is reachable only
  via the Datasets Server's pre-converted parquet shards. Treat as a yellow
  flag.
- ``loadable`` (bool) — **prefer datasets where this is ``True``.** Means
  the dataset is reachable via *some* path: either native parquet (no
  script) or HF auto-converted the script's output to parquet. When
  ``False``, the script needs deps HF can't install (e.g. ``conllu``,
  custom audio decoders) and ``aii_hf_datasets__download_datasets`` will
  fail — pick a different candidate.

**Parameters:**

`--query` (optional)
- Search query string
- Example: `--query "sentiment analysis"`

`--limit` (optional)
- Maximum number of results (default: 5)

`--tags` (optional)
- Filter by tags (comma-separated)
- Format: `category:value`
- Examples: `language:en`, `task_categories:text-classification`

`--sort` (optional)
- Sort by field: `downloads`, `likes` (default: downloads)

**Tips:**
- Search displays full dataset metadata
- Use tags to filter: `--tags "language:en,task_categories:translation"`

---

### Preview HuggingFace Dataset (aii_hf_preview_datasets.py)

Inspect a specific dataset - shows metadata AND sample rows.

**Example input:**
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-hf-datasets" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_hf_preview_datasets.py openai/gsm8k --num-rows 5
```

**Parallel execution (multiple datasets):**

IMPORTANT: Use full python path with GNU parallel:
```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-hf-datasets" && \
export PY="$SKILL_DIR/../.ability_client_venv/bin/python" && \
export S="$SKILL_DIR/scripts/aii_hf_preview_datasets.py" && \
parallel -j 10 -k --group --will-cite '$PY $S {} --num-rows 3' ::: 'openai/gsm8k' 'imdb' 'squad'
```

**Example output:**
```
============================================================
Dataset: openai/gsm8k
============================================================
Downloads: 425,109 | Likes: 1,102

Description: GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality
linguistically diverse grade school math word problems...

Configs: main, socratic

--- Sample Rows (train) ---
Columns: question, answer

Row 1:
  question: Natalia sold clips to 48 of her friends in April...
  answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May...
```

**Parameters:**

`dataset_id` (required, positional)
- HuggingFace dataset ID
- Examples: `openai/gsm8k`, `glue`, `imdb`

`--config` (optional)
- Dataset configuration/subset name
- Auto-detects first config if not specified

`--split` (optional)
- Split to preview (default: `train`)

`--num-rows` (optional)
- Number of sample rows (default: 5, max: 20)

**Tips:**
- Use after search to verify data structure
- Streaming mode - doesn't download full dataset

---

### Download HuggingFace Dataset (aii_hf_download_datasets.py)

Download datasets and save to files.

**Example input:**
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-hf-datasets" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_hf_download_datasets.py openai/gsm8k --config main --split train
```

**Parallel execution (multiple datasets):**

IMPORTANT: Use full python path with GNU parallel. Use `eval {}` pattern when datasets need different flags (e.g. `--config`):
```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-hf-datasets" && \
export PY="$SKILL_DIR/../.ability_client_venv/bin/python" && \
export S="$SKILL_DIR/scripts/aii_hf_download_datasets.py" && \
parallel -j 10 -k --group --will-cite 'eval {}' ::: '$PY $S openai/gsm8k --config main --split train' '$PY $S imdb --split train' '$PY $S squad --split train'
```

**Example output:**
```
Downloaded: openai/gsm8k

  train:
    Rows: 7,473
    Preview: temp/datasets/preview_openai_gsm8k_main_train.json
    Mini: temp/datasets/mini_openai_gsm8k_main_train.json
    Full: temp/datasets/full_openai_gsm8k_main_train.json
```

**Parameters:**

`dataset_id` (required, positional)
- HuggingFace dataset ID
- Examples: `openai/gsm8k`, `imdb`

`--config` (optional)
- Dataset configuration/subset name
- Use preview to see available configs

`--split` (optional)
- Specific split to load (e.g., `train`, `test`)
- If not specified, loads all splits

`--output-dir` (optional)
- Output directory (default: `temp/datasets/`)

**Output files (auto-saved):**
1. **Preview**: `preview_{dataset}_{split}.json` - 3 truncated rows - **READ THIS** for quick inspection
2. **Mini**: `mini_{dataset}_{split}.json` - 3 full rows - for development/testing
3. **Full**: `full_{dataset}_{split}.json` - All rows - **DO NOT READ directly** - use as input path for code

**Tips:**
- Only read preview file directly with Read tool
- Mini and full are input paths for processing code

**If the script fails** with a connection error (ability server not running): create a local `.venv`, install server deps from `server_requirements.txt` into it, then import the `@aii_ability` function from the script and call it directly — bypassing the server:
```bash
uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r "$SKILL_DIR/scripts/server_requirements.txt"
```
````

### [4] SKILL-INPUT — aii-python · 2026-06-22 05:08:41 UTC

The agent loaded the **aii-python** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-python
description: Python coding standards for experiment and evaluation scripts. Covers environment setup, logging, error handling, and code structure.
---

## Environment Setup

- Python 3.12+
- **NEVER use `pip` or `.venv/bin/pip`** — they are not installed. Use `uv` for ALL package operations:
  ```bash
  uv venv .venv --python=3.12
  source .venv/bin/activate  # or: .venv/bin/python script.py
  uv pip install pandas loguru  # NOT: pip install
  ```
- Create `.toml` file with dependencies, create uv `.venv` and activate it
- NO inline dependencies (no `# /// script` headers)

## Logging

Use `loguru` for all logging. Add a file sink alongside stdout.

```python
from loguru import logger
import sys

logger.remove()  # Remove default handler
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")
```

Rules:
- Log every major step (data loading, processing start/end, results)
- If applicable, log every LLM API call input and output
- Truncate long outputs in logs (add truncation logic for potentially large strings)
- Use `logger.error()` in except blocks (traceback auto-captured)

## Error Handling

- Wrap major operations in try/except blocks
- Use `@logger.catch(reraise=True)` decorator on main functions — without `reraise=True`, the script exits 0 even on uncaught exceptions, hiding failures from downstream consumers
- Use explicit exception types, not bare `except:`
- Never silently swallow exceptions — always log them

```python
@logger.catch(reraise=True)
def main():
    try:
        data = load_data(path)
    except FileNotFoundError:
        logger.error("Data file not found")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON in data file")
        raise
```

## Code Structure

- Use `pathlib.Path` for file operations: `Path("data/input.json").read_text()` not `open(...).read()`
- Use type hints for function signatures
- Use keyword arguments for functions with more than 4 parameters
- No hardcoded paths — derive from script location or accept as arguments

## Script Pattern

Standard pattern for experiment/evaluation scripts:

```python
#!/usr/bin/env python3
"""Brief description of what this script does."""

from loguru import logger
from pathlib import Path
import json
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def main():
    # Load data
    data_path = Path("full_data_out.json")
    logger.info(f"Loading data from {data_path}")
    data = json.loads(data_path.read_text())
    logger.info(f"Loaded {len(data['examples'])} examples")

    # Process
    results = []
    for i, example in enumerate(data["examples"]):
        try:
            result = process(example)
            results.append(result)
        except Exception:
            logger.error(f"Failed on example {i}")
            continue

    # Save output
    output = {"examples": results}
    Path("method_out.json").write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {len(results)} results")

if __name__ == "__main__":
    main()
```
````

### [5] SKILL-INPUT — aii-json · 2026-06-22 05:08:41 UTC

The agent loaded the **aii-json** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-json
description: JSON validation and formatting toolkit. Validate JSON files against schemas for experiment pipelines, and generate full/mini/preview versions of JSON datasets. Use for validating pipeline outputs, checking schema compliance, or creating size-optimized JSON variants.
---

## Contents

- Validating JSON (schema validation against experiment schemas)
- Formatting JSON (generate full/mini/preview versions)

**IMPORTANT - Parallel execution:** GNU `parallel` subshells do NOT inherit `source activate`. Use `export` for variables and **single-quoted** command templates so parallel's subshells can resolve them:
```
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
```

---

## Validating JSON

Validate JSON files against predefined schemas for experiment-based hypothesis selection, data collection, solution generation, and evaluation.

### Quick Start

1. Read the schema spec you need to adhere to (e.g., `schemas/exp_eval_sol_out.json`)
2. Create your output file following that schema structure
3. Validate:

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_json_validate_schema.py --format exp_eval_sol_out --file /path/to/eval_out.json
```

### Script: aii_json_validate_schema.py

**Example input:**
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_json_validate_schema.py --format exp_eval_sol_out --file /tmp/eval_out.json
```

**Parallel execution (multiple validations):**

IMPORTANT: When validating multiple files, use GNU parallel instead of separate Bash tool calls:
```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
export PY="$SKILL_DIR/../.ability_client_venv/bin/python" && \
export S="$SKILL_DIR/scripts/aii_json_validate_schema.py" && \
parallel -j 50 -k --group --will-cite '$PY $S --format {1} --file {2}' ::: 'exp_sel_data_out' 'exp_gen_sol_out' 'exp_eval_sol_out' :::+ '/tmp/full_data_out.json' '/tmp/method_out.json' '/tmp/eval_out.json'
```

**Example output (success):**
```
Validating: aii_json_validate_schema.py
Format: exp_eval_sol_out

✓ Validation PASSED
```

**Example output (failure):**
```
Validating: aii_json_validate_schema.py
Format: exp_sel_data_out

✗ Validation FAILED

Errors:
  Path: datasets → 0 → examples → 0
  Error: 'output' is a required property
  Validator: required
```

**Parameters:**

`--format` (required)
- Format type to validate against
- Determines which schema to use

`--file` (required)
- Path to JSON file to validate
- Must be valid JSON
- **Always pass an absolute path.** Relative paths resolve from the
  ability server's CWD (typically ``/ai-inventor/aii_server``), not from
  your agent workspace, so ``data_out/x.json`` will silently look in the
  wrong directory and fail with "Could not load JSON file". The validate
  endpoint also accepts a ``workspace_dir`` arg if you need to keep a
  relative path — pass your workspace path there.

**Tips:**
- Fix errors in your JSON and rerun validation until it passes

### Schema Files

Schemas are stored in `.claude/skills/aii-json/schemas/`:

**Hypothesis Selection & Evaluation:**
- `sel_hypo_out.json` - Hypothesis Selection output (all hypotheses with selected flags)
- `feasibility_eval_all.json` - All hypotheses with feasibility scores
- `feasibility_eval_top.json` - Top 5 most feasible hypotheses
- `novelty_research_one.json` - Single hypothesis novelty research arguments with citations
- `novelty_eval_all.json` - All hypotheses with novelty scores
- `novelty_eval_top.json` - Single best selected hypothesis

**Experiment Pipeline:**
- `exp_sel_data_out.json` - Experiment Data Selection format
- `exp_gen_sol_out.json` - Experiment Solution Generation format
- `exp_eval_sol_out.json` - Experiment Solution Evaluation format

---

## Formatting JSON

Generate three size-optimized versions of a JSON file for efficient development and preview:
- **full**: Identical to original (all data)
- **mini**: First 3 items only (for quick testing)
- **preview**: Mini + all strings truncated to 200 chars (for quick inspection)

### Quick Start

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_json_format_mini_preview.py --input method_out.json
```

### Script: aii_json_format_mini_preview.py

**Example input:**
```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_json_format_mini_preview.py --input method_out.json
```

**Parallel execution (multiple files):**

IMPORTANT: When formatting multiple files, use GNU parallel instead of separate Bash tool calls:
```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-json" && \
export PY="$SKILL_DIR/../.ability_client_venv/bin/python" && \
export S="$SKILL_DIR/scripts/aii_json_format_mini_preview.py" && \
parallel -j 50 -k --group --will-cite '$PY $S --input {}' ::: 'full_data_out.json' 'method_out.json' 'eval_out.json'
```

**Example output:**
```
Generated 3 versions:
  Full (50 items): /path/to/full_method_out.json
  Mini (3 items): /path/to/mini_method_out.json
  Preview (3 items, truncated): /path/to/preview_method_out.json
```

**Parameters:**

`--input` (required)
- Path to input JSON file
- Must have a top-level array
- Example: `method_out.json`, `full_data_out.json`

`--output-dir` (optional)
- Output directory for generated files
- Default: same directory as input file
- Files are prefixed with `full_`, `mini_`, `preview_`

**Output Files:**

All three files use the same base name with different prefixes:
- `full_{basename}.json` - Complete dataset (identical to original)
- `mini_{basename}.json` - First 3 array items only
- `preview_{basename}.json` - First 3 items with strings truncated to 200 chars

**Tips:**
- Input JSON must have a top-level array structure
- String truncation is recursive (applies to nested objects and arrays)
- Use preview files for quick inspection without reading large datasets
- Use mini files for developing/testing code before running on full dataset

**If the script fails** with a connection error (ability server not running): create a local `.venv`, install server deps from `server_requirements.txt` into it, then import the `@aii_ability` function from the script and call it directly — bypassing the server:
```bash
uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r "$SKILL_DIR/scripts/server_requirements.txt"
```
````

### [6] SYSTEM-USER prompt · 2026-06-22 05:20:33 UTC

```
YOUR PREVIOUS SESSION WAS INTERRUPTED: A single operation exceeded the 720s message timeout. Each individual operation must complete within 720s. Do NOT mock, skip, or compromise your execution — still do the real work. Try to make operations run faster if possible. If a command genuinely takes longer than 720s, split it into sequential parts that each complete within the time limit.

CONTINUE FOLLOWING THESE INSTRUCTIONS:

<workspace>
Your workspace: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/`:
GOOD: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/file.py`, `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>
<user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>
<artifact_plan>
id: gen_plan_dataset_1_idx2
type: dataset
title: Short Text Datasets for EVT-MinHash Near-Duplicate Detection Evaluation
summary: >-
  Collect and standardize 3+ short text datasets (tweets, SMS, news headlines) with 10K+ documents each, containing 10-100
  words per document, for evaluating MinHash-based near-duplicate detection with EVT-based confidence intervals.
runpod_compute_profile: gpu
ideal_dataset_criteria: >-
  Datasets must contain short text documents (10-100 words each) suitable for near-duplicate detection evaluation. Ideal datasets:
  (1) Twitter/tweet datasets with 10K+ tweets of 10-50 words, (2) SMS/messaging datasets with 10K+ messages of 5-30 words,
  (3) News headlines datasets with 10K+ headlines of 5-20 words. Documents should be real user-generated content (not synthetic).
  Text should be in English or easily translatable. Dataset size must be <300MB when downloaded. Must have clear text field
  and ability to assign unique doc_id. Shingle count (k=3 character shingles) should range from 8-95 for 10-100 word documents.
dataset_search_plan: "PHASE 1: Search HuggingFace Hub for candidate datasets\n\n1. Search for tweet datasets:\n   - Query:\
  \ 'tweet' → filter for datasets with 'text' field and short documents\n   - Specific candidates: 'tweet_eval', 'cardiffnlp/tweet_sentiment_multilingual',\
  \ 'twitter_sentiment'\n   - Preview each to verify: (a) text field exists, (b) average word count 10-100, (c) 10K+ rows\n\
  \n2. Search for SMS/messaging datasets:\n   - Query: 'sms' or 'text messaging' → look for SMS spam or messaging corpora\n\
  \   - Specific candidates: 'sms_spam_collection', 'uci/sms_spam', search for 'NUS SMS Corpus'\n   - Preview to verify text\
  \ length and volume\n\n3. Search for news headlines datasets:\n   - Query: 'news headlines' or 'ag_news' or 'news titles'\n\
  \   - Specific candidates: 'ag_news' (use only titles), 'bbc_news_summary', 'news_category_dataset'\n   - Preview to verify\
  \ titles/headlines are 5-20 words\n\nPHASE 2: Web search for additional datasets (if HuggingFace insufficient)\n\n1. Search\
  \ for 'Tweet dataset CSV download' or 'SMS dataset UCI'\n2. Check UCI ML repository for SMS Spam Collection\n3. Check Kaggle\
  \ for 'Twitter sentiment' or 'SMS spam' datasets\n4. Search for 'news headlines dataset' from sources like BBC, Reuters\n\
  \nPHASE 3: Download and preview promising candidates\n\n1. For each candidate dataset:\n   a. Run aii_hf_preview_datasets.py\
  \ to inspect structure\n   b. Check: number of rows, text field name, sample text lengths\n   c. Calculate word count distribution\
  \ on sample\n   d. Verify <300MB total size\n\n2. Select top 3-5 datasets meeting criteria:\n   - 10K+ documents\n   - 10-100\
  \ words per document (80th percentile)\n   - Clear text field\n   - Loadable via HuggingFace or direct download\n\nPHASE\
  \ 4: Download full datasets and standardize\n\n1. Download each selected dataset using aii_hf_download_datasets.py\n2. Standardize\
  \ to JSON format with REQUIRED fields:\n   {\n     \"doc_id\": \"unique_string\",\n     \"text\": \"document text content\"\
  ,\n     \"source\": \"dataset_name\",\n     \"length\": {\n       \"words\": 45,\n       \"shingles_k3\": 128\n     }\n\
  \   }\n\n3. For each dataset, compute:\n   a. doc_id: Use dataset's ID field or generate sequential IDs\n   b. text: Extract\
  \ and clean text field (remove HTML, normalize whitespace)\n   c. source: Dataset name (e.g., 'tweet_eval', 'sms_spam')\n\
  \   d. length.words: Count words (split on whitespace, filter empty)\n   e. length.shingles_k3: Count unique 3-character\
  \ shingles\n\n4. Filter documents:\n   - Keep only documents with 10-100 words\n   - If >50% of documents outside range,\
  \ try different preprocessing or reject dataset\n\nPHASE 5: Create dataset splits\n\nFor each dataset, create three files:\n\
  1. preview_{dataset}.json: 3 rows with truncated text (100 chars max)\n2. mini_{dataset}.json: 100 random rows (full text)\n\
  3. full_{dataset}.json: All rows (full text)\n\nPHASE 6: Validation and fallback\n\n1. Validate each dataset:\n   - Check\
  \ total document count (warn if <5K, error if <1K)\n   - Verify JSON schema matches required fields\n   - Check word count\
  \ distribution (histogram)\n   - Sample 10 documents and manually verify quality\n\n2. If insufficient datasets found:\n\
  \   FALLBACK OPTION A: Generate synthetic near-duplicates\n   - Take 1K base documents from any dataset\n   - Create near-duplicates\
  \ by: (a) deleting 10-30% of words, (b) replacing synonyms, (c) reordering sentences\n   - This creates pairs with known\
  \ Jaccard similarity\n   \n   FALLBACK OPTION B: Sample from larger datasets\n   - Use 'bookcorpus' or 'wikipedia' but extract\
  \ short paragraphs/sentences\n   - Filter for length 10-100 words\n\n   FALLBACK OPTION C: Combine multiple small datasets\n\
  \   - Merge 3-4 smaller datasets (each 2-5K docs) to reach 10K+\n\n3. Final deliverable: 3+ datasets in data_out.json format\n\
  \   - Merge all datasets into single data_out.json with field 'dataset_name' to identify source\n   - Include metadata:\
  \ total_docs, avg_words, source_datasets\n\nEXECUTION NOTES:\n- Use aii-hf-datasets skill for HuggingFace datasets (search/preview/download)\n\
  - Use aii-web-tools skill for web searches if HuggingFace insufficient\n- Use aii-json skill to validate output JSON schema\n\
  - Use aii-file-size-limit skill to check output files don't exceed limits\n- All processing in Python 3.12 with pandas/numpy\
  \ for data manipulation\n- Maximum 300MB total dataset size (enforced by preview step)\n- Budget: $0 LLM API calls (no OpenRouter\
  \ needed for dataset collection)\n\nSPECIFIC DATASET RECOMMENDATIONS (start here):\n\n1. tweet_eval (HuggingFace: tweet_eval)\n\
  \   - Contains tweets for emotion/sentiment tasks\n   - ~10K tweets per subset, ~10-30 words each\n   - Preview: aii_hf_preview_datasets.py\
  \ tweet_eval --config emotion --num-rows 10\n\n2. sms_spam_collection (HuggingFace: uci/sms_spam or sms_spam_collection)\n\
  \   - SMS messages, ~5-30 words each\n   - ~5.5K messages (may need to combine with another dataset)\n   - Preview: aii_hf_preview_datasets.py\
  \ uci/sms_spam --num-rows 10\n\n3. ag_news (HuggingFace: ag_news)\n   - News articles with titles (use only titles field)\n\
  \   - ~120K articles, titles are 5-15 words\n   - Preview: aii_hf_preview_datasets.py ag_news --num-rows 10\n\n4. bbc_news_summary\
  \ (HuggingFace: bbc_news_summary)\n   - BBC news headlines and summaries\n   - Use headlines (10-20 words)\n   - Preview:\
  \ aii_hf_preview_datasets.py bbc_news_summary --num-rows 10\n\n5. cardiffnlp/tweet_sentiment_multilingual (HuggingFace)\n\
  \   - Multilingual tweets, filter for English\n   - ~12K tweets, 10-40 words\n   - Preview: aii_hf_preview_datasets.py cardiffnlp/tweet_sentiment_multilingual\
  \ --num-rows 10"
target_num_datasets: 3
</artifact_plan>



<available_resources>
<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>

<skills>
Skills are self-contained capabilities with instructions, context, and tools.

- aii-web-tools: Web search (Serper), page/PDF fetch as markdown, regex grep over page/PDF text
- aii-semscholar-bib: Batch-fetch BibTeX from Semantic Scholar
- aii-openrouter-llms: Search and call 300+ LLMs via OpenRouter
- aii-hf-datasets: Search, preview, download HuggingFace datasets
- aii-owid-datasets: Search and load Our World in Data tables
- aii-lean: Compile/verify Lean 4 code, Mathlib search, tactic suggestions
- aii-image-gen: Generate/edit images via Gemini 3 Pro Image (Nano Banana Pro)
- aii-json: Validate JSON against schemas, generate mini/preview variants
- aii-paper-writing: Academic paper structure, bibliography, citations
- aii-paper-to-latex: Assemble LaTeX papers and compile to PDF
- aii-parallel-computing: GPU acceleration, CPU parallelism, async I/O
- aii-python: Python coding standards for experiment scripts
- aii-use-hardware: Detect CPU/RAM/GPU, memory-safe processing
- aii-long-running-tasks: Gradual scaling pattern for long-running tasks
- aii-colab: Google Colab runtime constraints for notebooks
- aii-file-size-limit: Check and split oversized output files
- aii-handbook-multi-llm-agents: Multi-LLM agent orchestration patterns
</skills>
</available_resources>

<available_data_sources>
Use the sources appropriate to your task. Read the relevant skill file BEFORE using each source.

- **HuggingFace Hub** (HF) — ML datasets (NLP, vision, tabular, benchmarks)
- **Our World in Data** (OWID) — Global statistics (energy, health, economics, environment, demographics)
- **Alternate methods** — Python/shell (sklearn.datasets, openml, direct URL, APIs, etc.)

If the plan specifies a source or one fits better, use it.
You may combine sources. Use web search (aii-web-tools skill) to research candidates (background, papers, provenance) — NOT to find/download datasets.
</available_data_sources>

<available_domain_handbooks>
If your domain has a handbook, read the relevant skill file BEFORE working on that domain.

- **Multi-LLM Agents** — dataset selection, evaluation metrics, agent orchestration patterns
</available_domain_handbooks>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<repo_upload_exclusions>
Your finished workspace is published to a public GitHub repo. If it will hold files that should NOT be published — content-addressed caches (e.g. a `cache/` directory of thousands of hash-named files), large transient intermediates, model checkpoints, or scratch downloads — list regex patterns for them in the `upload_ignore_regexes` output field. Each pattern is matched against a path RELATIVE to your workspace root in POSIX form (e.g. `(^|/)cache/`, `(^|/)checkpoints/`). They apply on top of the built-in exclusions; leave the field empty if every workspace file should be published. Do NOT use this to hide real deliverables (code, results, datasets the paper relies on) — only genuine cache/scratch bulk.
</repo_upload_exclusions>

IMPORTANT: Your final response should be at most 300 characters long.

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Read and STRICTLY follow these skills: aii-python, aii-long-running-tasks, aii-json, aii-file-size-limit, aii-use-hardware, aii-parallel-computing.
TODO 2. Read skill files for your data sources (see <available_data_sources>) and domain handbook if applicable (see <available_domain_handbooks>). Based on plan and context, decide which source(s) to use. Include everything specified in the artifact plan, but you may also collect additional relevant data beyond what's listed. Run 24 diverse searches across chosen source(s) — BROAD, GENERAL terms, not very specific. Parallelize where supported.
TODO 3. Identify the 12 most promising datasets. IMPORTANT: Only consider datasets under 300MB. Preview/inspect sample rows for each candidate. Parallelize previews.
TODO 4. Research each candidate BEFORE choosing which to download. For each, search the web (aii-web-tools skill): dataset name, papers citing it, original source/task, popularity. Red flags: no search results, no papers, anonymized features (F1, F2...), <100 downloads, no documentation. Green flags: papers using it, clear documentation, meaningful features, established benchmark. Also consider: will features/structure allow meaningful evaluation of the planned method?
TODO 5. Decide which to KEEP vs DISCARD. Look for: clear structure, relevant fields, quality examples matching requirements, confirmed provenance. Determine which 6 datasets have the most suitable data. Download and save to `temp/datasets/`. Parallelize downloads.
</todos>

Build and evaluate a simple MinHash near-duplicate detector for short text documents.
```

### [7] SYSTEM-USER prompt · 2026-06-22 05:23:43 UTC

````
<workspace>
Your workspace: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/`:
GOOD: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/file.py`, `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>
<user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>
<artifact_plan>
id: gen_plan_dataset_1_idx2
type: dataset
title: Short Text Datasets for EVT-MinHash Near-Duplicate Detection Evaluation
summary: >-
  Collect and standardize 3+ short text datasets (tweets, SMS, news headlines) with 10K+ documents each, containing 10-100
  words per document, for evaluating MinHash-based near-duplicate detection with EVT-based confidence intervals.
runpod_compute_profile: gpu
ideal_dataset_criteria: >-
  Datasets must contain short text documents (10-100 words each) suitable for near-duplicate detection evaluation. Ideal datasets:
  (1) Twitter/tweet datasets with 10K+ tweets of 10-50 words, (2) SMS/messaging datasets with 10K+ messages of 5-30 words,
  (3) News headlines datasets with 10K+ headlines of 5-20 words. Documents should be real user-generated content (not synthetic).
  Text should be in English or easily translatable. Dataset size must be <300MB when downloaded. Must have clear text field
  and ability to assign unique doc_id. Shingle count (k=3 character shingles) should range from 8-95 for 10-100 word documents.
dataset_search_plan: "PHASE 1: Search HuggingFace Hub for candidate datasets\n\n1. Search for tweet datasets:\n   - Query:\
  \ 'tweet' → filter for datasets with 'text' field and short documents\n   - Specific candidates: 'tweet_eval', 'cardiffnlp/tweet_sentiment_multilingual',\
  \ 'twitter_sentiment'\n   - Preview each to verify: (a) text field exists, (b) average word count 10-100, (c) 10K+ rows\n\
  \n2. Search for SMS/messaging datasets:\n   - Query: 'sms' or 'text messaging' → look for SMS spam or messaging corpora\n\
  \   - Specific candidates: 'sms_spam_collection', 'uci/sms_spam', search for 'NUS SMS Corpus'\n   - Preview to verify text\
  \ length and volume\n\n3. Search for news headlines datasets:\n   - Query: 'news headlines' or 'ag_news' or 'news titles'\n\
  \   - Specific candidates: 'ag_news' (use only titles), 'bbc_news_summary', 'news_category_dataset'\n   - Preview to verify\
  \ titles/headlines are 5-20 words\n\nPHASE 2: Web search for additional datasets (if HuggingFace insufficient)\n\n1. Search\
  \ for 'Tweet dataset CSV download' or 'SMS dataset UCI'\n2. Check UCI ML repository for SMS Spam Collection\n3. Check Kaggle\
  \ for 'Twitter sentiment' or 'SMS spam' datasets\n4. Search for 'news headlines dataset' from sources like BBC, Reuters\n\
  \nPHASE 3: Download and preview promising candidates\n\n1. For each candidate dataset:\n   a. Run aii_hf_preview_datasets.py\
  \ to inspect structure\n   b. Check: number of rows, text field name, sample text lengths\n   c. Calculate word count distribution\
  \ on sample\n   d. Verify <300MB total size\n\n2. Select top 3-5 datasets meeting criteria:\n   - 10K+ documents\n   - 10-100\
  \ words per document (80th percentile)\n   - Clear text field\n   - Loadable via HuggingFace or direct download\n\nPHASE\
  \ 4: Download full datasets and standardize\n\n1. Download each selected dataset using aii_hf_download_datasets.py\n2. Standardize\
  \ to JSON format with REQUIRED fields:\n   {\n     \"doc_id\": \"unique_string\",\n     \"text\": \"document text content\"\
  ,\n     \"source\": \"dataset_name\",\n     \"length\": {\n       \"words\": 45,\n       \"shingles_k3\": 128\n     }\n\
  \   }\n\n3. For each dataset, compute:\n   a. doc_id: Use dataset's ID field or generate sequential IDs\n   b. text: Extract\
  \ and clean text field (remove HTML, normalize whitespace)\n   c. source: Dataset name (e.g., 'tweet_eval', 'sms_spam')\n\
  \   d. length.words: Count words (split on whitespace, filter empty)\n   e. length.shingles_k3: Count unique 3-character\
  \ shingles\n\n4. Filter documents:\n   - Keep only documents with 10-100 words\n   - If >50% of documents outside range,\
  \ try different preprocessing or reject dataset\n\nPHASE 5: Create dataset splits\n\nFor each dataset, create three files:\n\
  1. preview_{dataset}.json: 3 rows with truncated text (100 chars max)\n2. mini_{dataset}.json: 100 random rows (full text)\n\
  3. full_{dataset}.json: All rows (full text)\n\nPHASE 6: Validation and fallback\n\n1. Validate each dataset:\n   - Check\
  \ total document count (warn if <5K, error if <1K)\n   - Verify JSON schema matches required fields\n   - Check word count\
  \ distribution (histogram)\n   - Sample 10 documents and manually verify quality\n\n2. If insufficient datasets found:\n\
  \   FALLBACK OPTION A: Generate synthetic near-duplicates\n   - Take 1K base documents from any dataset\n   - Create near-duplicates\
  \ by: (a) deleting 10-30% of words, (b) replacing synonyms, (c) reordering sentences\n   - This creates pairs with known\
  \ Jaccard similarity\n   \n   FALLBACK OPTION B: Sample from larger datasets\n   - Use 'bookcorpus' or 'wikipedia' but extract\
  \ short paragraphs/sentences\n   - Filter for length 10-100 words\n\n   FALLBACK OPTION C: Combine multiple small datasets\n\
  \   - Merge 3-4 smaller datasets (each 2-5K docs) to reach 10K+\n\n3. Final deliverable: 3+ datasets in data_out.json format\n\
  \   - Merge all datasets into single data_out.json with field 'dataset_name' to identify source\n   - Include metadata:\
  \ total_docs, avg_words, source_datasets\n\nEXECUTION NOTES:\n- Use aii-hf-datasets skill for HuggingFace datasets (search/preview/download)\n\
  - Use aii-web-tools skill for web searches if HuggingFace insufficient\n- Use aii-json skill to validate output JSON schema\n\
  - Use aii-file-size-limit skill to check output files don't exceed limits\n- All processing in Python 3.12 with pandas/numpy\
  \ for data manipulation\n- Maximum 300MB total dataset size (enforced by preview step)\n- Budget: $0 LLM API calls (no OpenRouter\
  \ needed for dataset collection)\n\nSPECIFIC DATASET RECOMMENDATIONS (start here):\n\n1. tweet_eval (HuggingFace: tweet_eval)\n\
  \   - Contains tweets for emotion/sentiment tasks\n   - ~10K tweets per subset, ~10-30 words each\n   - Preview: aii_hf_preview_datasets.py\
  \ tweet_eval --config emotion --num-rows 10\n\n2. sms_spam_collection (HuggingFace: uci/sms_spam or sms_spam_collection)\n\
  \   - SMS messages, ~5-30 words each\n   - ~5.5K messages (may need to combine with another dataset)\n   - Preview: aii_hf_preview_datasets.py\
  \ uci/sms_spam --num-rows 10\n\n3. ag_news (HuggingFace: ag_news)\n   - News articles with titles (use only titles field)\n\
  \   - ~120K articles, titles are 5-15 words\n   - Preview: aii_hf_preview_datasets.py ag_news --num-rows 10\n\n4. bbc_news_summary\
  \ (HuggingFace: bbc_news_summary)\n   - BBC news headlines and summaries\n   - Use headlines (10-20 words)\n   - Preview:\
  \ aii_hf_preview_datasets.py bbc_news_summary --num-rows 10\n\n5. cardiffnlp/tweet_sentiment_multilingual (HuggingFace)\n\
  \   - Multilingual tweets, filter for English\n   - ~12K tweets, 10-40 words\n   - Preview: aii_hf_preview_datasets.py cardiffnlp/tweet_sentiment_multilingual\
  \ --num-rows 10"
target_num_datasets: 3
</artifact_plan>



<available_resources>
<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>

<skills>
Skills are self-contained capabilities with instructions, context, and tools.

- aii-web-tools: Web search (Serper), page/PDF fetch as markdown, regex grep over page/PDF text
- aii-semscholar-bib: Batch-fetch BibTeX from Semantic Scholar
- aii-openrouter-llms: Search and call 300+ LLMs via OpenRouter
- aii-hf-datasets: Search, preview, download HuggingFace datasets
- aii-owid-datasets: Search and load Our World in Data tables
- aii-lean: Compile/verify Lean 4 code, Mathlib search, tactic suggestions
- aii-image-gen: Generate/edit images via Gemini 3 Pro Image (Nano Banana Pro)
- aii-json: Validate JSON against schemas, generate mini/preview variants
- aii-paper-writing: Academic paper structure, bibliography, citations
- aii-paper-to-latex: Assemble LaTeX papers and compile to PDF
- aii-parallel-computing: GPU acceleration, CPU parallelism, async I/O
- aii-python: Python coding standards for experiment scripts
- aii-use-hardware: Detect CPU/RAM/GPU, memory-safe processing
- aii-long-running-tasks: Gradual scaling pattern for long-running tasks
- aii-colab: Google Colab runtime constraints for notebooks
- aii-file-size-limit: Check and split oversized output files
- aii-handbook-multi-llm-agents: Multi-LLM agent orchestration patterns
</skills>
</available_resources>

<available_data_sources>
Use the sources appropriate to your task. Read the relevant skill file BEFORE using each source.

- **HuggingFace Hub** (HF) — ML datasets (NLP, vision, tabular, benchmarks)
- **Our World in Data** (OWID) — Global statistics (energy, health, economics, environment, demographics)
- **Alternate methods** — Python/shell (sklearn.datasets, openml, direct URL, APIs, etc.)

If the plan specifies a source or one fits better, use it.
You may combine sources. Use web search (aii-web-tools skill) to research candidates (background, papers, provenance) — NOT to find/download datasets.
</available_data_sources>

<available_domain_handbooks>
If your domain has a handbook, read the relevant skill file BEFORE working on that domain.

- **Multi-LLM Agents** — dataset selection, evaluation metrics, agent orchestration patterns
</available_domain_handbooks>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<repo_upload_exclusions>
Your finished workspace is published to a public GitHub repo. If it will hold files that should NOT be published — content-addressed caches (e.g. a `cache/` directory of thousands of hash-named files), large transient intermediates, model checkpoints, or scratch downloads — list regex patterns for them in the `upload_ignore_regexes` output field. Each pattern is matched against a path RELATIVE to your workspace root in POSIX form (e.g. `(^|/)cache/`, `(^|/)checkpoints/`). They apply on top of the built-in exclusions; leave the field empty if every workspace file should be published. Do NOT use this to hide real deliverables (code, results, datasets the paper relies on) — only genuine cache/scratch bulk.
</repo_upload_exclusions>

IMPORTANT: Your final response should be at most 300 characters long.

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. For the top 6 datasets, create data.py (uv inline script) that: loads from temp/datasets/, standardizes to exp_sel_data_out.json schema (aii-json skill), extracts all examples per dataset, handles domain requirements, saves to full_data_out.json.

Each data ROW must be a separate example — do NOT create one example per dataset or per fold. Each data point (row, sample, instance) = one example. 500 rows → 500 examples. The output is GROUPED BY DATASET:
```json
{
  "datasets": [
    {
      "dataset": "iris",
      "examples": [
        {"input": "...", "output": "...", "metadata_fold": 2, "metadata_feature_names": [...]},
        ...
      ]
    },
    {
      "dataset": "adult_census",
      "examples": [...]
    }
  ]
}
```
Per-example required fields:
- `input`: input features/text (tabular: JSON string of feature values)
- `output`: target/label (as string)
Per-example optional metadata via `metadata_<name>` fields (flat, not nested object):
- `metadata_fold`: fold assignment (int), `metadata_feature_names`: feature name list, `metadata_task_type`: "classification"/"regression", `metadata_n_classes`: number of classes, `metadata_row_index`: original row index, etc.
Do NOT use `split`, `dataset`, or `context` as per-example fields. Dataset name goes at the group level, metadata goes in `metadata_*` fields.
TODO 2. Run 'uv run data.py' and fix errors. Validate full_data_out.json against exp_sel_data_out.json schema (aii-json skill) — fix errors. Generate preview, mini, full versions with aii-json skill's format script.
TODO 3. Read preview to inspect examples. Choose THE BEST 3 DATASETS based on domain requirements and artifact objective. Be very attentive to meticulously and exhaustively fix any errors in your code.
</todos>
````

### [8] SYSTEM-USER prompt · 2026-06-22 05:25:41 UTC

````
<workspace>
Your workspace: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/`:
GOOD: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/file.py`, `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>
<user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>
<artifact_plan>
id: gen_plan_dataset_1_idx2
type: dataset
title: Short Text Datasets for EVT-MinHash Near-Duplicate Detection Evaluation
summary: >-
  Collect and standardize 3+ short text datasets (tweets, SMS, news headlines) with 10K+ documents each, containing 10-100
  words per document, for evaluating MinHash-based near-duplicate detection with EVT-based confidence intervals.
runpod_compute_profile: gpu
ideal_dataset_criteria: >-
  Datasets must contain short text documents (10-100 words each) suitable for near-duplicate detection evaluation. Ideal datasets:
  (1) Twitter/tweet datasets with 10K+ tweets of 10-50 words, (2) SMS/messaging datasets with 10K+ messages of 5-30 words,
  (3) News headlines datasets with 10K+ headlines of 5-20 words. Documents should be real user-generated content (not synthetic).
  Text should be in English or easily translatable. Dataset size must be <300MB when downloaded. Must have clear text field
  and ability to assign unique doc_id. Shingle count (k=3 character shingles) should range from 8-95 for 10-100 word documents.
dataset_search_plan: "PHASE 1: Search HuggingFace Hub for candidate datasets\n\n1. Search for tweet datasets:\n   - Query:\
  \ 'tweet' → filter for datasets with 'text' field and short documents\n   - Specific candidates: 'tweet_eval', 'cardiffnlp/tweet_sentiment_multilingual',\
  \ 'twitter_sentiment'\n   - Preview each to verify: (a) text field exists, (b) average word count 10-100, (c) 10K+ rows\n\
  \n2. Search for SMS/messaging datasets:\n   - Query: 'sms' or 'text messaging' → look for SMS spam or messaging corpora\n\
  \   - Specific candidates: 'sms_spam_collection', 'uci/sms_spam', search for 'NUS SMS Corpus'\n   - Preview to verify text\
  \ length and volume\n\n3. Search for news headlines datasets:\n   - Query: 'news headlines' or 'ag_news' or 'news titles'\n\
  \   - Specific candidates: 'ag_news' (use only titles), 'bbc_news_summary', 'news_category_dataset'\n   - Preview to verify\
  \ titles/headlines are 5-20 words\n\nPHASE 2: Web search for additional datasets (if HuggingFace insufficient)\n\n1. Search\
  \ for 'Tweet dataset CSV download' or 'SMS dataset UCI'\n2. Check UCI ML repository for SMS Spam Collection\n3. Check Kaggle\
  \ for 'Twitter sentiment' or 'SMS spam' datasets\n4. Search for 'news headlines dataset' from sources like BBC, Reuters\n\
  \nPHASE 3: Download and preview promising candidates\n\n1. For each candidate dataset:\n   a. Run aii_hf_preview_datasets.py\
  \ to inspect structure\n   b. Check: number of rows, text field name, sample text lengths\n   c. Calculate word count distribution\
  \ on sample\n   d. Verify <300MB total size\n\n2. Select top 3-5 datasets meeting criteria:\n   - 10K+ documents\n   - 10-100\
  \ words per document (80th percentile)\n   - Clear text field\n   - Loadable via HuggingFace or direct download\n\nPHASE\
  \ 4: Download full datasets and standardize\n\n1. Download each selected dataset using aii_hf_download_datasets.py\n2. Standardize\
  \ to JSON format with REQUIRED fields:\n   {\n     \"doc_id\": \"unique_string\",\n     \"text\": \"document text content\"\
  ,\n     \"source\": \"dataset_name\",\n     \"length\": {\n       \"words\": 45,\n       \"shingles_k3\": 128\n     }\n\
  \   }\n\n3. For each dataset, compute:\n   a. doc_id: Use dataset's ID field or generate sequential IDs\n   b. text: Extract\
  \ and clean text field (remove HTML, normalize whitespace)\n   c. source: Dataset name (e.g., 'tweet_eval', 'sms_spam')\n\
  \   d. length.words: Count words (split on whitespace, filter empty)\n   e. length.shingles_k3: Count unique 3-character\
  \ shingles\n\n4. Filter documents:\n   - Keep only documents with 10-100 words\n   - If >50% of documents outside range,\
  \ try different preprocessing or reject dataset\n\nPHASE 5: Create dataset splits\n\nFor each dataset, create three files:\n\
  1. preview_{dataset}.json: 3 rows with truncated text (100 chars max)\n2. mini_{dataset}.json: 100 random rows (full text)\n\
  3. full_{dataset}.json: All rows (full text)\n\nPHASE 6: Validation and fallback\n\n1. Validate each dataset:\n   - Check\
  \ total document count (warn if <5K, error if <1K)\n   - Verify JSON schema matches required fields\n   - Check word count\
  \ distribution (histogram)\n   - Sample 10 documents and manually verify quality\n\n2. If insufficient datasets found:\n\
  \   FALLBACK OPTION A: Generate synthetic near-duplicates\n   - Take 1K base documents from any dataset\n   - Create near-duplicates\
  \ by: (a) deleting 10-30% of words, (b) replacing synonyms, (c) reordering sentences\n   - This creates pairs with known\
  \ Jaccard similarity\n   \n   FALLBACK OPTION B: Sample from larger datasets\n   - Use 'bookcorpus' or 'wikipedia' but extract\
  \ short paragraphs/sentences\n   - Filter for length 10-100 words\n\n   FALLBACK OPTION C: Combine multiple small datasets\n\
  \   - Merge 3-4 smaller datasets (each 2-5K docs) to reach 10K+\n\n3. Final deliverable: 3+ datasets in data_out.json format\n\
  \   - Merge all datasets into single data_out.json with field 'dataset_name' to identify source\n   - Include metadata:\
  \ total_docs, avg_words, source_datasets\n\nEXECUTION NOTES:\n- Use aii-hf-datasets skill for HuggingFace datasets (search/preview/download)\n\
  - Use aii-web-tools skill for web searches if HuggingFace insufficient\n- Use aii-json skill to validate output JSON schema\n\
  - Use aii-file-size-limit skill to check output files don't exceed limits\n- All processing in Python 3.12 with pandas/numpy\
  \ for data manipulation\n- Maximum 300MB total dataset size (enforced by preview step)\n- Budget: $0 LLM API calls (no OpenRouter\
  \ needed for dataset collection)\n\nSPECIFIC DATASET RECOMMENDATIONS (start here):\n\n1. tweet_eval (HuggingFace: tweet_eval)\n\
  \   - Contains tweets for emotion/sentiment tasks\n   - ~10K tweets per subset, ~10-30 words each\n   - Preview: aii_hf_preview_datasets.py\
  \ tweet_eval --config emotion --num-rows 10\n\n2. sms_spam_collection (HuggingFace: uci/sms_spam or sms_spam_collection)\n\
  \   - SMS messages, ~5-30 words each\n   - ~5.5K messages (may need to combine with another dataset)\n   - Preview: aii_hf_preview_datasets.py\
  \ uci/sms_spam --num-rows 10\n\n3. ag_news (HuggingFace: ag_news)\n   - News articles with titles (use only titles field)\n\
  \   - ~120K articles, titles are 5-15 words\n   - Preview: aii_hf_preview_datasets.py ag_news --num-rows 10\n\n4. bbc_news_summary\
  \ (HuggingFace: bbc_news_summary)\n   - BBC news headlines and summaries\n   - Use headlines (10-20 words)\n   - Preview:\
  \ aii_hf_preview_datasets.py bbc_news_summary --num-rows 10\n\n5. cardiffnlp/tweet_sentiment_multilingual (HuggingFace)\n\
  \   - Multilingual tweets, filter for English\n   - ~12K tweets, 10-40 words\n   - Preview: aii_hf_preview_datasets.py cardiffnlp/tweet_sentiment_multilingual\
  \ --num-rows 10"
target_num_datasets: 3
</artifact_plan>



<available_resources>
<software_constraints>
- Python only implementation
- Python standard library and all popular PyPI packages available (numpy, pandas, scikit-learn, scipy, matplotlib, requests, etc.)
- Local parallelism encouraged: multiprocessing, asyncio, threading — see aii-parallel-computing skill
- LLM API calls must go through OpenRouter only (no direct OpenAI, Anthropic, etc.)
- **HARD LIMIT**: Maximum $10 USD total spend on LLM API calls (OpenRouter). Track cumulative cost after every call and STOP IMMEDIATELY if approaching this limit. Never exceed this budget under any circumstances.
</software_constraints>

<skills>
Skills are self-contained capabilities with instructions, context, and tools.

- aii-web-tools: Web search (Serper), page/PDF fetch as markdown, regex grep over page/PDF text
- aii-semscholar-bib: Batch-fetch BibTeX from Semantic Scholar
- aii-openrouter-llms: Search and call 300+ LLMs via OpenRouter
- aii-hf-datasets: Search, preview, download HuggingFace datasets
- aii-owid-datasets: Search and load Our World in Data tables
- aii-lean: Compile/verify Lean 4 code, Mathlib search, tactic suggestions
- aii-image-gen: Generate/edit images via Gemini 3 Pro Image (Nano Banana Pro)
- aii-json: Validate JSON against schemas, generate mini/preview variants
- aii-paper-writing: Academic paper structure, bibliography, citations
- aii-paper-to-latex: Assemble LaTeX papers and compile to PDF
- aii-parallel-computing: GPU acceleration, CPU parallelism, async I/O
- aii-python: Python coding standards for experiment scripts
- aii-use-hardware: Detect CPU/RAM/GPU, memory-safe processing
- aii-long-running-tasks: Gradual scaling pattern for long-running tasks
- aii-colab: Google Colab runtime constraints for notebooks
- aii-file-size-limit: Check and split oversized output files
- aii-handbook-multi-llm-agents: Multi-LLM agent orchestration patterns
</skills>
</available_resources>

<available_data_sources>
Use the sources appropriate to your task. Read the relevant skill file BEFORE using each source.

- **HuggingFace Hub** (HF) — ML datasets (NLP, vision, tabular, benchmarks)
- **Our World in Data** (OWID) — Global statistics (energy, health, economics, environment, demographics)
- **Alternate methods** — Python/shell (sklearn.datasets, openml, direct URL, APIs, etc.)

If the plan specifies a source or one fits better, use it.
You may combine sources. Use web search (aii-web-tools skill) to research candidates (background, papers, provenance) — NOT to find/download datasets.
</available_data_sources>

<available_domain_handbooks>
If your domain has a handbook, read the relevant skill file BEFORE working on that domain.

- **Multi-LLM Agents** — dataset selection, evaluation metrics, agent orchestration patterns
</available_domain_handbooks>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<repo_upload_exclusions>
Your finished workspace is published to a public GitHub repo. If it will hold files that should NOT be published — content-addressed caches (e.g. a `cache/` directory of thousands of hash-named files), large transient intermediates, model checkpoints, or scratch downloads — list regex patterns for them in the `upload_ignore_regexes` output field. Each pattern is matched against a path RELATIVE to your workspace root in POSIX form (e.g. `(^|/)cache/`, `(^|/)checkpoints/`). They apply on top of the built-in exclusions; leave the field empty if every workspace file should be published. Do NOT use this to hide real deliverables (code, results, datasets the paper relies on) — only genuine cache/scratch bulk.
</repo_upload_exclusions>

IMPORTANT: Your final response should be at most 300 characters long.

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Update data.py to only include the chosen 3 datasets and generate full_data_out.json. Re-run to generate full_data_out.json. Validate output format with aii-json skill and fix any errors. Generate full, mini, and preview versions with aii-json skill's format script using `--input full_data_out.json` (creates full_full_data_out.json, mini_full_data_out.json, preview_full_data_out.json — rename to full_data_out.json, mini_data_out.json, preview_data_out.json).
TODO 2. Verify full_data_out.json, preview_data_out.json, and mini_data_out.json exist in your workspace (see <workspace>) and contain correct data.
TODO 3. Apply aii-file-size-limit skill's file size check procedure (100MB limit) to full_data_out.json.
TODO 4. Ensure a `pyproject.toml` exists in your workspace with ALL dependencies pinned to the exact versions installed in your .venv (run `.venv/bin/pip freeze` to get them). This is required for reproducibility. The [project] section must include name, version, requires-python, and a dependencies list with pinned versions (e.g. `numpy==2.0.2`, not `numpy>=2.0`).
</todos>

---

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "DatasetExpectedFiles": {
      "description": "All expected output files from dataset artifact.",
      "properties": {
        "script": {
          "description": "Path to data.py script. Example: 'data.py'",
          "title": "Script",
          "type": "string"
        },
        "datasets": {
          "description": "Dataset file groups \u2014 one per dataset, each with full/mini/preview variants",
          "items": {
            "$ref": "#/$defs/DatasetFileSet"
          },
          "title": "Datasets",
          "type": "array"
        }
      },
      "required": [
        "script",
        "datasets"
      ],
      "title": "DatasetExpectedFiles",
      "type": "object"
    },
    "DatasetFileSet": {
      "description": "One dataset's three required output variants.",
      "properties": {
        "full": {
          "description": "Full dataset JSON file(s). Single file or split files. Example: ['full_data_out.json'] or ['full_data_out/full_data_out_1.json', 'full_data_out/full_data_out_2.json']",
          "items": {
            "type": "string"
          },
          "title": "Full",
          "type": "array"
        },
        "mini": {
          "description": "Mini dataset JSON file path (3 examples). Example: 'mini_data_out.json'",
          "title": "Mini",
          "type": "string"
        },
        "preview": {
          "description": "Preview dataset JSON file path (10 examples). Example: 'preview_data_out.json'",
          "title": "Preview",
          "type": "string"
        }
      },
      "required": [
        "full",
        "mini",
        "preview"
      ],
      "title": "DatasetFileSet",
      "type": "object"
    }
  },
  "description": "Dataset artifact \u2014 structured output + file metadata.\n\nFinds, evaluates, and prepares datasets for research experiments.\nProduces data.py and full_data_out.json files.",
  "properties": {
    "title": {
      "default": "",
      "description": "Descriptive title (roughly 30-90 characters). Must describe content, NOT a status message.",
      "maxLength": 90,
      "minLength": 30,
      "title": "Title",
      "type": "string"
    },
    "layman_summary": {
      "default": "",
      "description": "One-sentence plain-language summary of what this artifact does, accessible to non-experts. Used only in the per-artifact README, not in downstream prompts.",
      "maxLength": 250,
      "minLength": 80,
      "title": "Layman Summary",
      "type": "string"
    },
    "summary": {
      "default": "",
      "description": "Summary for downstream artifacts: what this artifact provides",
      "maxLength": 5000,
      "minLength": 500,
      "title": "Summary",
      "type": "string"
    },
    "out_expected_files": {
      "$ref": "#/$defs/DatasetExpectedFiles",
      "description": "All output files you created. Must include data.py script plus dataset file groups (full/mini/preview variants)."
    },
    "upload_ignore_regexes": {
      "description": "Regex patterns for workspace paths that must NOT be published to the GitHub repo, matched against each file's path relative to this artifact's workspace root (POSIX form, e.g. 'cache/abc.json'). Applied ON TOP OF the deploy step's built-in exclusions. Use this for executor-specific caches, large transient intermediates, or content-addressed blob stores (e.g. a cache/ dir of thousands of hash-named files) that would bloat the repo. Examples: ['(^|/)cache/', '(^|/)\\\\.weight_cache/', '(^|/)checkpoints/']. Leave empty if every workspace file should be published.",
      "items": {
        "type": "string"
      },
      "title": "Upload Ignore Regexes",
      "type": "array"
    }
  },
  "required": [
    "out_expected_files"
  ],
  "title": "DatasetArtifact",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/.sdk_openhands_agent_struct_out.json`.
````
