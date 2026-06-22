# gen_plan_dataset_1 — test_idea

> Phase: `invention_loop` · round 1 · `gen_plan`
> Run: `run_EqcgJR2naF4b` — Why Extreme Value Theory Fails for MinHash Confidence Intervals on Short Text (and What to Do Instead)
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_plan_dataset_1` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 04:52:16 UTC

````
<hypothesis>
kind: hypothesis
title: >-
  EVT-MinHash: Extreme Value Theory for Statistically Principled Confidence Intervals in MinHash-Based Near-Duplicate Detection
hypothesis: >-
  For short text documents, the distribution of MinHash signature values follows a Gumbel distribution (Extreme Value Type
  I) because MinHash computes the minimum of hash values across shingles. By modeling this distribution explicitly using Extreme
  Value Theory (EVT), we can derive closed-form confidence intervals and hypothesis tests for Jaccard similarity estimates
  that are more accurate and computationally efficient than bootstrap-based methods, especially in the small-sample regime
  where short documents produce few shingles.
motivation: >-
  Near-duplicate detection using MinHash is fundamentally a statistical estimation problem, yet current approaches lack statistically
  principled uncertainty quantification. For short text documents (tweets, SMS, titles), the small number of shingles leads
  to high-variance Jaccard estimates, and practitioners must rely on heuristics or computationally expensive bootstrap methods
  to assess confidence. The core mechanism of MinHash—computing the minimum of random hash values—directly connects to Extreme
  Value Theory, which provides the asymptotic distribution of minima. This cross-disciplinary transfer from mathematical statistics
  to near-duplicate detection enables rigorous uncertainty quantification with minimal computational overhead, making it practical
  for large-scale deduplication pipelines where short text is common.
assumptions:
- >-
  Hash functions produce approximately uniformly distributed values in [0,1] for randomly selected shingles
- >-
  The number of shingles in short text documents (10-100) is sufficient for the asymptotic Gumbel distribution to provide
  a reasonable approximation
- >-
  Shingles within a document can be treated as approximately independent samples for the purpose of modeling the distribution
  of minima
- >-
  The computational overhead of EVT-based confidence intervals is negligible compared to the MinHash computation itself
investigation_approach: >-
  1. Derive the theoretical connection between MinHash and the Gumbel distribution by modeling MinHash values as order statistics
  of independent random variables. 2. Develop closed-form expressions for confidence intervals of Jaccard similarity estimates
  using the delta method applied to the Gumbel distribution parameters. 3. Implement EVT-MinHash and compare against standard
  MinHash with bootstrap confidence intervals on short text datasets (tweets, SMS, news headlines). 4. Evaluate on three axes:
  (a) coverage probability (do 95% CI actually contain true Jaccard 95% of the time?), (b) interval width (are EVT intervals
  tighter?), and (c) computational cost. 5. Test the hypothesis that EVT-based intervals achieve better coverage with lower
  computational cost than bootstrap alternatives for documents with fewer than 100 shingles.
success_criteria: >-
  The EVT-based confidence intervals should: (1) Achieve coverage probability within 2% of the nominal level (e.g., 95±2%
  for nominal 95% CI) on held-out test data, (2) Be computationally cheaper than bootstrap (no resampling required), and (3)
  Produce narrower intervals than bootstrap with the same coverage for short documents (<50 shingles). Additionally, a hypothesis
  test derived from the EVT framework should correctly control false positive rate at the specified alpha level when comparing
  document pairs.
related_works:
- >-
  LSHBloom (arXiv 2411.04257, 2024): Proposes architectural improvements to MinHashLSH using Bloom filters for internet-scale
  deduplication. Our work differs fundamentally by focusing on the statistical properties of the MinHash estimator itself
  (using EVT) rather than indexing optimizations. LSHBloom does not provide confidence intervals or hypothesis tests for Jaccard
  estimates.
- >-
  Sampling-Based Estimation of Jaccard Containment and Similarity (arXiv 2507.10019, 2025): Derives likelihood models and
  posterior error bounds for Jaccard estimation from samples. Our work differs by using Extreme Value Theory to model the
  distribution of MinHash values directly, providing closed-form confidence intervals without requiring posterior sampling.
  The core mechanism (EVT vs. Bayesian posterior) is fundamentally different.
- >-
  Debiasing FracMinHash and deriving confidence intervals (biorxiv 2022.01.11.475870): Derives confidence intervals for FracMinHash
  in bioinformatics applications. Our work differs by (a) using EVT/Gumbel distribution as the theoretical foundation rather
  than the normal approximation, (b) focusing specifically on short text documents where the small-sample properties matter
  most, and (c) providing a hypothesis testing framework, not just confidence intervals.
- >-
  SetSketch: Filling the Gap between MinHash and HyperLogLog (VLDB 2021): Uses Gumbel distribution for cardinality estimation,
  not for Jaccard similarity confidence intervals. Our work transfers this EVT insight to the similarity estimation problem,
  which requires different theoretical treatment since we are modeling the joint distribution of two MinHash signatures, not
  a single set's cardinality.
- >-
  Weighted MinHash algorithms (arXiv 1811.04633): Focuses on extending MinHash to weighted sets using various sampling strategies.
  Our work is orthogonal—we use standard MinHash but add EVT-based uncertainty quantification, which is compatible with weighted
  variants but addresses a different problem (uncertainty vs. weighting).
inspiration: >-
  The hypothesis draws from three cross-domain inspirations: (1) Extreme Value Theory from mathematical statistics—the Fisher-Tippett-Gnedenko
  theorem states that the minimum of i.i.d. random variables converges to a Gumbel distribution, which directly applies to
  MinHash's minimum operation. (2) Order statistics from probability theory—MinHash signatures are essentially order statistics
  (the minimum) of transformed random variables. (3) The delta method from statistical theory—used to propagate uncertainty
  from the Gumbel parameters to the Jaccard similarity estimate. The key insight is that MinHash is not just a heuristic algorithm
  but a statistical estimator whose sampling distribution can be characterized analytically using EVT.
terms:
- term: MinHash
  definition: >-
    A locality-sensitive hashing algorithm that estimates Jaccard similarity between sets by comparing the minimum hash values
    of their elements. It works by applying multiple hash functions to each set and keeping only the minimum hash value for
    each function, producing a signature that preserves similarity.
- term: Jaccard Similarity
  definition: >-
    A similarity coefficient between two sets A and B defined as |A∩B|/|A∪B|, ranging from 0 (disjoint sets) to 1 (identical
    sets). For near-duplicate detection, higher Jaccard similarity indicates greater overlap between documents.
- term: Extreme Value Theory (EVT)
  definition: >-
    A branch of statistics dealing with the extreme deviations from the median of probability distributions. The Fisher-Tippett-Gnedenko
    theorem states that the minimum (or maximum) of i.i.d. random variables converges to one of three distributions: Gumbel,
    Fréchet, or Weibull. For MinHash (which uses minima of continuous random variables), the Gumbel distribution applies.
- term: Gumbel Distribution
  definition: >-
    A probability distribution of extreme values (minima or maxima) also known as the Extreme Value Type I distribution. It
    has location parameter μ and scale parameter σ. The CDF is F(x) = exp(-exp(-(x-μ)/σ)). For MinHash, the distribution of
    minimum hash values across shingles converges to Gumbel as the number of shingles increases.
- term: Shingle
  definition: >-
    A contiguous subsequence of tokens or characters from a document, used as a fingerprint. A k-shingle is a subsequence
    of length k. For example, the 3-shingles of 'abcde' are 'abc', 'bcd', 'cde'. The set of all shingles from a document represents
    its content for similarity comparison.
- term: Confidence Interval
  definition: >-
    A range of values derived from sample data that is likely to contain the true population parameter with a specified probability
    (confidence level). For MinHash, a 95% confidence interval for Jaccard similarity means that if we repeated the sampling
    process many times, 95% of such intervals would contain the true Jaccard similarity.
- term: Near-Duplicate Detection
  definition: >-
    The task of identifying pairs of documents that are approximately identical (e.g., different versions of the same article,
    copied content with minor modifications). Unlike exact deduplication, near-duplicate detection tolerates small differences
    in wording, formatting, or structure.
summary: >-
  By modeling MinHash signature values using Extreme Value Theory (Gumbel distribution), we can derive closed-form confidence
  intervals and hypothesis tests for Jaccard similarity estimates that are more accurate and computationally efficient than
  bootstrap methods, especially for short text documents with few shingles.
</hypothesis>

<artifact_direction>
Make this direction concrete and actionable. Keep the same type and respect dependencies.

id: dataset_iter1_dir2
type: dataset
objective: >-
  Collect and prepare short text datasets suitable for evaluating MinHash-based near-duplicate detection with EVT-based confidence
  intervals
approach: >-
  Search for and download datasets containing short text documents: (1) Twitter/tweet datasets from HuggingFace or similar
  sources, (2) SMS/messaging datasets, (3) News headlines datasets. Target documents with 10-100 words (short texts where
  small-sample properties matter). Standardize to JSON format with fields: doc_id, text, source, length (in words/shingles).
  Create full/mini/preview splits. Aim for 10K+ documents to enable robust evaluation.
depends_on: []
</artifact_direction>



<instructions>
YOUR ROLE: Write a detailed PLAN for the artifact. A separate executor agent runs the actual artifact later.

You are a PLANNER, not an executor. Your output is a plan that tells the executor what to do and how.
Do NOT execute the artifact itself — a separate agent handles that. Your job is to plan it so well that the executor can follow your plan step by step.

You CAN and SHOULD: search the web, read papers, and explore library docs to make your plan concrete.
You CANNOT run shell commands or scripts — code execution is disabled. Research via web tools only.

Do NOT do the executor's job: don't download datasets, don't implement code, don't run experiments, don't write proofs, don't compute evaluations.

<artifact_executor_scope>
IMPORTANT: Each artifact executor has a focused prompt that guides it to do ONE thing well. It will NOT perform tasks outside its scope — assigning the wrong work to the wrong artifact type wastes an iteration. Match the task to the right executor.

DATASET executor scope:
  Output: data_out.json with rows of {input, output, metadata_fold, ...} — raw data only, no derived computations
  DOES: Download/generate datasets, analyze candidates to pick the best ones, standardize to JSON schema (features, labels, folds, metadata), validate schema, split into full/mini/preview
  DOES NOT: Run experiments, train models, compute derived statistics (PID/MI/correlations/synergy matrices) as final output
  If you need to COMPUTE something from data (synergy matrices, MI scores, timing benchmarks), use an EXPERIMENT artifact instead
</artifact_executor_scope>

<artifact_planning_rules>
DATASET:
- Plan for REAL third-party datasets (HuggingFace, Kaggle, direct-download URLs) — downloadable within time and size constraints
- Describe dataset criteria (domain, size, format) — executors find exact sources, but you can suggest candidates or search directions
- ALWAYS prefer real datasets over synthetic. Synthetic is a LAST RESORT only when no suitable real data exists
</artifact_planning_rules>

<compute_profiles>
Choose the compute profile this artifact needs for execution.
Available profiles for dataset artifacts:
  - gpu: 1x NVIDIA RTX A4500, 20GB VRAM, 7 vCPUs, 29GB RAM — ML training, CUDA, large models (fallback: GPUs cheap→expensive: 2000 Ada → A4000 → 4000 Ada → L4 → 4090 → 5090)
  - cpu_heavy: 4 vCPUs, 32GB RAM — large datasets, memory-intensive processing (fallback: CPUs cheap→expensive, then GPU hosts cheap→expensive (all ≥32GB RAM))

Set runpod_compute_profile to one of these exact tier names.
</compute_profiles>
GOOD PLANS: specific, actionable, consider failure scenarios, build on the suggested approach.
BAD PLANS: vague hand-waving, ignoring the suggested approach, missing critical executor details.
</instructions><user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>

---

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_plan/gen_plan_dataset_1/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "description": "Plan for a DATASET artifact.",
  "properties": {
    "title": {
      "description": "Short title for the plan",
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "default": "",
      "description": "Brief summary",
      "title": "Summary",
      "type": "string"
    },
    "runpod_compute_profile": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": "cpu_light",
      "description": "Compute tier for execution \u2014 pick from the available profiles list (e.g., 'gpu', 'cpu_heavy', 'cpu_light'). Only used in RunPod mode.",
      "title": "Runpod Compute Profile"
    },
    "ideal_dataset_criteria": {
      "description": "What makes an ideal dataset for this purpose - size, format, content requirements",
      "title": "Ideal Dataset Criteria",
      "type": "string"
    },
    "dataset_search_plan": {
      "description": "Step-by-step plan for finding/creating this dataset - sources to check, fallback options",
      "title": "Dataset Search Plan",
      "type": "string"
    },
    "target_num_datasets": {
      "description": "How many individual datasets should be delivered. Count each dataset separately, not collections \u2014 a benchmark suite of N datasets counts as N. This controls how broadly the executor searches, so setting it too low will under-collect.",
      "title": "Target Num Datasets",
      "type": "integer"
    }
  },
  "required": [
    "title",
    "ideal_dataset_criteria",
    "dataset_search_plan",
    "target_num_datasets"
  ],
  "title": "DatasetPlan",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_plan/gen_plan_dataset_1/.sdk_openhands_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-06-22 04:52:16 UTC

```
Build and evaluate a simple MinHash near-duplicate detector for short text documents.
```

### [3] SKILL-INPUT — aii-web-tools · 2026-06-22 04:52:42 UTC

The agent loaded the **aii-web-tools** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-web-tools
description: "Web research toolkit: web search (Serper/Google), web page fetch as markdown (HTML and PDF), and regex grep over full page/PDF text. Use whenever a task needs to search the web, read a page, mine a paper/PDF, verify citations, or extract exact quotes, numbers, or methodology from a URL."
---

## Web tools

You have three web capabilities: **search**, **fetch**, and **grep** (exact
regex extraction over a full page or PDF).

**Pick where they come from, in this order:**

1. **If you have built-in `WebSearch` / `WebFetch` tools, PREFER those over the
   scripts below.** They may be **deferred tools** (listed by name but with
   schemas not yet loaded) — if so, call `ToolSearch("select:WebSearch,WebFetch")`
   ONCE to load them, then use them normally. Do not skip them just because they
   need that one extra load step; they are the preferred path. Pair them with the
   `aii_web_tools__fetch_grep` script below when you need exact text / numbers /
   methodology that a summary would miss, or when reading a PDF.
2. **Only if you have NO built-in `WebSearch` / `WebFetch`** (e.g. the OpenHands
   backend), use the scripts in this skill (below). They are our own
   implementations — Serper.dev for search, html2text + PyMuPDF for fetch, and
   regex grep over the full document text. They work without any built-in web
   tools.

Workflow either way: **search** (discover) → **fetch** (read for the gist) →
**grep** (pull exact details / read PDFs).

---

## Running the scripts

Run every script with the skill's pre-provisioned interpreter (it already has
`requests`, `html2text`, `pymupdf`, `python-dotenv`). Set `PY` once:

```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-web-tools"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
```

### 1. Search the web (Serper.dev / Google)

```bash
$PY "$SKILL_DIR/scripts/aii_fast_web_search.py" --query "neuro-symbolic FOL translation LLM" --max-results 10
```

Returns ranked title / URL / snippet lines. Use it first to scan the
landscape; snippets are for discovery only — fetch a page before judging it.

### 2. Fetch a page as markdown (HTML or PDF)

```bash
$PY "$SKILL_DIR/scripts/aii_fast_web_fetch.py" fetch --url "https://arxiv.org/abs/2303.11366" --max-chars 10000
```

`--max-chars` caps output (default 10000); `--char-offset N` pages further in.
Handles PDFs transparently via PyMuPDF.

### 3. Grep a page or PDF (exact regex extraction)

```bash
$PY "$SKILL_DIR/scripts/aii_fast_web_fetch.py" grep --url "https://arxiv.org/pdf/2303.11366" --pattern "verbal reinforcement" --max-matches 20 --context-chars 200
```

Returns only the matching sections with surrounding context — the right tool
for exact numbers, table values, methodology, or long PDFs where a summary
would lose the detail. `-i` for case-insensitive.

**Parallelize** independent searches/fetches in one turn; only sequence a
fetch after the search that produced its URL.

---

## Notes

- The scripts call our ability server. If a script prints
  `Ability service not available`, the server is down — say so rather than
  silently improvising a different search method.
- Do **not** hand-roll your own `requests`/scraping for search when these
  tools are available: Serper returns clean Google results and the fetch/grep
  scripts already handle HTML, PDFs, and encoding.
````

### [4] SKILL-INPUT — aii-hf-datasets · 2026-06-22 04:53:15 UTC

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

### [5] SYSTEM-USER prompt · 2026-06-22 05:04:42 UTC

````
YOUR PREVIOUS SESSION WAS INTERRUPTED: A single operation exceeded the 720s message timeout. Each individual operation must complete within 720s. Do NOT mock, skip, or compromise your execution — still do the real work. Try to make operations run faster if possible. If a command genuinely takes longer than 720s, split it into sequential parts that each complete within the time limit.

Last messages before interruption:
  - [agent_response]: I'll research suitable short text datasets for evaluating MinHash near-duplicate detection. Let me search for specific datasets that match the criteria (10-100 words, tweets/SMS/headlines).
  - [agent_tool_call] invoke_skill: Tool: invoke_skill
{
  "name": "aii-hf-datasets"
}
  - [agent_tool_call] invoke_skill: Tool: invoke_skill
{
  "name": "aii-web-tools"
}
  - [agent_tool_result] invoke_skill: Tool: invoke_skill
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "## Contents\n\n- Workflow (3-phase dataset discovery)\n- Scripts (Search, Preview, Download)\n\n**IMPORTANT - Parallel execution:** GNU `parallel` subshells do NOT inherit `source activate`. Use `export` for var
  - [agent_tool_result] invoke_skill: Tool: invoke_skill
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "## Web tools\n\nYou have three web capabilities: **search**, **fetch**, and **grep** (exact\nregex extraction over a full page or PDF).\n\n**Pick where they come from, in this order:**\n\n1. **If you have built-
  - [agent_think]: The user wants me to create a detailed plan for collecting short text datasets suitable for evaluating EVT-MinHash near-duplicate detection. I need to:

1. Research and identify specific datasets that match the criteria:
   - Short text documents (10-100 words)
   - Types: Twitter/tweets, SMS/messag
  - [agent_response]: I'll research and create a detailed plan for collecting short text datasets suitable for evaluating EVT-MinHash near-duplicate detection. Let me first organize my thoughts on the specific datasets and search strategy needed.
  - [agent_tool_call] think: Tool: think
{
  "thought": "I need to create a detailed, actionable plan for collecting short text datasets for evaluating MinHash-based near-duplicate detection with EVT-based confidence intervals.\n\nKey requirements from the artifact direction:\n1. Short text datasets (10-100 words/documents)\n2.
  - [agent_tool_result] think: Tool: think
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "Your thought has been logged."}], "is_error": false, "kind": "ThinkObservation"}
  - [agent_think]: Let me organize my thinking and create a comprehensive plan for the dataset collection. I need to:

1. Search for specific datasets that match the criteria (short text: 10-100 words)
2. Target three types: tweets, SMS, news headlines
3. Aim for 10K+ documents
4. Standardize to JSON format with speci
  - [agent_response]: Now I'll create the detailed plan based on my research. Let me write the plan JSON file with specific, actionable steps for the executor.
  - [agent_tool_call] file_editor: Tool: file_editor
{
  "command": "create",
  "path": "/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_plan/gen_plan_dataset_1/.sdk_openhands_agent_struct_out.json",
  "file_text": "{\n  \"title\": \"Short Text Datasets for EVT-MinHash Near-Duplicate Detection Evaluation\",\n 
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "File created successfully at: /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_plan/gen_plan_dataset_1/.sdk_openhands_agent_struct_out.json"}], "is_error": false, "command": "create", "path
  - [agent_response]: Let me verify the file was created correctly and review its contents.
  - [agent_tool_call] file_editor: Tool: file_editor
{
  "command": "view",
  "path": "/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_plan/gen_plan_dataset_1/.sdk_openhands_agent_struct_out.json",
  "file_text": null,
  "old_str": null,
  "new_str": null,
  "insert_line": null,
  "view_range": null
}
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "Here's the result of running `cat -n` on /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_plan/gen_plan_dataset_1/.sdk_openhands_agent_struct_out.json:\n     1\t{\n     2\t  \"title\": \"Sh
  - [agent_think]: The file has been created successfully. Let me review it to make sure it meets all the requirements:

1. **Title**: ✓ "Short Text Datasets for EVT-MinHash Near-Duplicate Detection Evaluation"
2. **Summary**: ✓ Brief summary provided
3. **runpod_compute_profile**: ✓ "cpu_light" - appropriate for data
  - [agent_response]: The plan has been created successfully. Let me verify the JSON is valid and complete by checking its structure.
  - [agent_tool_call] finish: Tool: finish
{
  "message": "The detailed plan for collecting short text datasets has been created and saved to `.sdk_openhands_agent_struct_out.json`.\n\n## Plan Summary\n\n**Title**: Short Text Datasets for EVT-MinHash Near-Duplicate Detection Evaluation\n\n**Objective**: Collect and standardize 3
  - [agent_tool_result] finish: Tool: finish
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "The detailed plan for collecting short text datasets has been created and saved to `.sdk_openhands_agent_struct_out.json`.\n\n## Plan Summary\n\n**Title**: Short Text Datasets for EVT-MinHash Near-Duplicate Detection 

CONTINUE FOLLOWING THESE INSTRUCTIONS:

<hypothesis>
kind: hypothesis
title: >-
  EVT-MinHash: Extreme Value Theory for Statistically Principled Confidence Intervals in MinHash-Based Near-Duplicate Detection
hypothesis: >-
  For short text documents, the distribution of MinHash signature values follows a Gumbel distribution (Extreme Value Type
  I) because MinHash computes the minimum of hash values across shingles. By modeling this distribution explicitly using Extreme
  Value Theory (EVT), we can derive closed-form confidence intervals and hypothesis tests for Jaccard similarity estimates
  that are more accurate and computationally efficient than bootstrap-based methods, especially in the small-sample regime
  where short documents produce few shingles.
motivation: >-
  Near-duplicate detection using MinHash is fundamentally a statistical estimation problem, yet current approaches lack statistically
  principled uncertainty quantification. For short text documents (tweets, SMS, titles), the small number of shingles leads
  to high-variance Jaccard estimates, and practitioners must rely on heuristics or computationally expensive bootstrap methods
  to assess confidence. The core mechanism of MinHash—computing the minimum of random hash values—directly connects to Extreme
  Value Theory, which provides the asymptotic distribution of minima. This cross-disciplinary transfer from mathematical statistics
  to near-duplicate detection enables rigorous uncertainty quantification with minimal computational overhead, making it practical
  for large-scale deduplication pipelines where short text is common.
assumptions:
- >-
  Hash functions produce approximately uniformly distributed values in [0,1] for randomly selected shingles
- >-
  The number of shingles in short text documents (10-100) is sufficient for the asymptotic Gumbel distribution to provide
  a reasonable approximation
- >-
  Shingles within a document can be treated as approximately independent samples for the purpose of modeling the distribution
  of minima
- >-
  The computational overhead of EVT-based confidence intervals is negligible compared to the MinHash computation itself
investigation_approach: >-
  1. Derive the theoretical connection between MinHash and the Gumbel distribution by modeling MinHash values as order statistics
  of independent random variables. 2. Develop closed-form expressions for confidence intervals of Jaccard similarity estimates
  using the delta method applied to the Gumbel distribution parameters. 3. Implement EVT-MinHash and compare against standard
  MinHash with bootstrap confidence intervals on short text datasets (tweets, SMS, news headlines). 4. Evaluate on three axes:
  (a) coverage probability (do 95% CI actually contain true Jaccard 95% of the time?), (b) interval width (are EVT intervals
  tighter?), and (c) computational cost. 5. Test the hypothesis that EVT-based intervals achieve better coverage with lower
  computational cost than bootstrap alternatives for documents with fewer than 100 shingles.
success_criteria: >-
  The EVT-based confidence intervals should: (1) Achieve coverage probability within 2% of the nominal level (e.g., 95±2%
  for nominal 95% CI) on held-out test data, (2) Be computationally cheaper than bootstrap (no resampling required), and (3)
  Produce narrower intervals than bootstrap with the same coverage for short documents (<50 shingles). Additionally, a hypothesis
  test derived from the EVT framework should correctly control false positive rate at the specified alpha level when comparing
  document pairs.
related_works:
- >-
  LSHBloom (arXiv 2411.04257, 2024): Proposes architectural improvements to MinHashLSH using Bloom filters for internet-scale
  deduplication. Our work differs fundamentally by focusing on the statistical properties of the MinHash estimator itself
  (using EVT) rather than indexing optimizations. LSHBloom does not provide confidence intervals or hypothesis tests for Jaccard
  estimates.
- >-
  Sampling-Based Estimation of Jaccard Containment and Similarity (arXiv 2507.10019, 2025): Derives likelihood models and
  posterior error bounds for Jaccard estimation from samples. Our work differs by using Extreme Value Theory to model the
  distribution of MinHash values directly, providing closed-form confidence intervals without requiring posterior sampling.
  The core mechanism (EVT vs. Bayesian posterior) is fundamentally different.
- >-
  Debiasing FracMinHash and deriving confidence intervals (biorxiv 2022.01.11.475870): Derives confidence intervals for FracMinHash
  in bioinformatics applications. Our work differs by (a) using EVT/Gumbel distribution as the theoretical foundation rather
  than the normal approximation, (b) focusing specifically on short text documents where the small-sample properties matter
  most, and (c) providing a hypothesis testing framework, not just confidence intervals.
- >-
  SetSketch: Filling the Gap between MinHash and HyperLogLog (VLDB 2021): Uses Gumbel distribution for cardinality estimation,
  not for Jaccard similarity confidence intervals. Our work transfers this EVT insight to the similarity estimation problem,
  which requires different theoretical treatment since we are modeling the joint distribution of two MinHash signatures, not
  a single set's cardinality.
- >-
  Weighted MinHash algorithms (arXiv 1811.04633): Focuses on extending MinHash to weighted sets using various sampling strategies.
  Our work is orthogonal—we use standard MinHash but add EVT-based uncertainty quantification, which is compatible with weighted
  variants but addresses a different problem (uncertainty vs. weighting).
inspiration: >-
  The hypothesis draws from three cross-domain inspirations: (1) Extreme Value Theory from mathematical statistics—the Fisher-Tippett-Gnedenko
  theorem states that the minimum of i.i.d. random variables converges to a Gumbel distribution, which directly applies to
  MinHash's minimum operation. (2) Order statistics from probability theory—MinHash signatures are essentially order statistics
  (the minimum) of transformed random variables. (3) The delta method from statistical theory—used to propagate uncertainty
  from the Gumbel parameters to the Jaccard similarity estimate. The key insight is that MinHash is not just a heuristic algorithm
  but a statistical estimator whose sampling distribution can be characterized analytically using EVT.
terms:
- term: MinHash
  definition: >-
    A locality-sensitive hashing algorithm that estimates Jaccard similarity between sets by comparing the minimum hash values
    of their elements. It works by applying multiple hash functions to each set and keeping only the minimum hash value for
    each function, producing a signature that preserves similarity.
- term: Jaccard Similarity
  definition: >-
    A similarity coefficient between two sets A and B defined as |A∩B|/|A∪B|, ranging from 0 (disjoint sets) to 1 (identical
    sets). For near-duplicate detection, higher Jaccard similarity indicates greater overlap between documents.
- term: Extreme Value Theory (EVT)
  definition: >-
    A branch of statistics dealing with the extreme deviations from the median of probability distributions. The Fisher-Tippett-Gnedenko
    theorem states that the minimum (or maximum) of i.i.d. random variables converges to one of three distributions: Gumbel,
    Fréchet, or Weibull. For MinHash (which uses minima of continuous random variables), the Gumbel distribution applies.
- term: Gumbel Distribution
  definition: >-
    A probability distribution of extreme values (minima or maxima) also known as the Extreme Value Type I distribution. It
    has location parameter μ and scale parameter σ. The CDF is F(x) = exp(-exp(-(x-μ)/σ)). For MinHash, the distribution of
    minimum hash values across shingles converges to Gumbel as the number of shingles increases.
- term: Shingle
  definition: >-
    A contiguous subsequence of tokens or characters from a document, used as a fingerprint. A k-shingle is a subsequence
    of length k. For example, the 3-shingles of 'abcde' are 'abc', 'bcd', 'cde'. The set of all shingles from a document represents
    its content for similarity comparison.
- term: Confidence Interval
  definition: >-
    A range of values derived from sample data that is likely to contain the true population parameter with a specified probability
    (confidence level). For MinHash, a 95% confidence interval for Jaccard similarity means that if we repeated the sampling
    process many times, 95% of such intervals would contain the true Jaccard similarity.
- term: Near-Duplicate Detection
  definition: >-
    The task of identifying pairs of documents that are approximately identical (e.g., different versions of the same article,
    copied content with minor modifications). Unlike exact deduplication, near-duplicate detection tolerates small differences
    in wording, formatting, or structure.
summary: >-
  By modeling MinHash signature values using Extreme Value Theory (Gumbel distribution), we can derive closed-form confidence
  intervals and hypothesis tests for Jaccard similarity estimates that are more accurate and computationally efficient than
  bootstrap methods, especially for short text documents with few shingles.
</hypothesis>

<artifact_direction>
Make this direction concrete and actionable. Keep the same type and respect dependencies.

id: dataset_iter1_dir2
type: dataset
objective: >-
  Collect and prepare short text datasets suitable for evaluating MinHash-based near-duplicate detection with EVT-based confidence
  intervals
approach: >-
  Search for and download datasets containing short text documents: (1) Twitter/tweet datasets from HuggingFace or similar
  sources, (2) SMS/messaging datasets, (3) News headlines datasets. Target documents with 10-100 words (short texts where
  small-sample properties matter). Standardize to JSON format with fields: doc_id, text, source, length (in words/shingles).
  Create full/mini/preview splits. Aim for 10K+ documents to enable robust evaluation.
depends_on: []
</artifact_direction>



<instructions>
YOUR ROLE: Write a detailed PLAN for the artifact. A separate executor agent runs the actual artifact later.

You are a PLANNER, not an executor. Your output is a plan that tells the executor what to do and how.
Do NOT execute the artifact itself — a separate agent handles that. Your job is to plan it so well that the executor can follow your plan step by step.

You CAN and SHOULD: search the web, read papers, and explore library docs to make your plan concrete.
You CANNOT run shell commands or scripts — code execution is disabled. Research via web tools only.

Do NOT do the executor's job: don't download datasets, don't implement code, don't run experiments, don't write proofs, don't compute evaluations.

<artifact_executor_scope>
IMPORTANT: Each artifact executor has a focused prompt that guides it to do ONE thing well. It will NOT perform tasks outside its scope — assigning the wrong work to the wrong artifact type wastes an iteration. Match the task to the right executor.

DATASET executor scope:
  Output: data_out.json with rows of {input, output, metadata_fold, ...} — raw data only, no derived computations
  DOES: Download/generate datasets, analyze candidates to pick the best ones, standardize to JSON schema (features, labels, folds, metadata), validate schema, split into full/mini/preview
  DOES NOT: Run experiments, train models, compute derived statistics (PID/MI/correlations/synergy matrices) as final output
  If you need to COMPUTE something from data (synergy matrices, MI scores, timing benchmarks), use an EXPERIMENT artifact instead
</artifact_executor_scope>

<artifact_planning_rules>
DATASET:
- Plan for REAL third-party datasets (HuggingFace, Kaggle, direct-download URLs) — downloadable within time and size constraints
- Describe dataset criteria (domain, size, format) — executors find exact sources, but you can suggest candidates or search directions
- ALWAYS prefer real datasets over synthetic. Synthetic is a LAST RESORT only when no suitable real data exists
</artifact_planning_rules>

<compute_profiles>
Choose the compute profile this artifact needs for execution.
Available profiles for dataset artifacts:
  - gpu: 1x NVIDIA RTX A4500, 20GB VRAM, 7 vCPUs, 29GB RAM — ML training, CUDA, large models (fallback: GPUs cheap→expensive: 2000 Ada → A4000 → 4000 Ada → L4 → 4090 → 5090)
  - cpu_heavy: 4 vCPUs, 32GB RAM — large datasets, memory-intensive processing (fallback: CPUs cheap→expensive, then GPU hosts cheap→expensive (all ≥32GB RAM))

Set runpod_compute_profile to one of these exact tier names.
</compute_profiles>
GOOD PLANS: specific, actionable, consider failure scenarios, build on the suggested approach.
BAD PLANS: vague hand-waving, ignoring the suggested approach, missing critical executor details.
</instructions><user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>

---

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_plan/gen_plan_dataset_1/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "description": "Plan for a DATASET artifact.",
  "properties": {
    "title": {
      "description": "Short title for the plan",
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "default": "",
      "description": "Brief summary",
      "title": "Summary",
      "type": "string"
    },
    "runpod_compute_profile": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": "cpu_light",
      "description": "Compute tier for execution \u2014 pick from the available profiles list (e.g., 'gpu', 'cpu_heavy', 'cpu_light'). Only used in RunPod mode.",
      "title": "Runpod Compute Profile"
    },
    "ideal_dataset_criteria": {
      "description": "What makes an ideal dataset for this purpose - size, format, content requirements",
      "title": "Ideal Dataset Criteria",
      "type": "string"
    },
    "dataset_search_plan": {
      "description": "Step-by-step plan for finding/creating this dataset - sources to check, fallback options",
      "title": "Dataset Search Plan",
      "type": "string"
    },
    "target_num_datasets": {
      "description": "How many individual datasets should be delivered. Count each dataset separately, not collections \u2014 a benchmark suite of N datasets counts as N. This controls how broadly the executor searches, so setting it too low will under-collect.",
      "title": "Target Num Datasets",
      "type": "integer"
    }
  },
  "required": [
    "title",
    "ideal_dataset_criteria",
    "dataset_search_plan",
    "target_num_datasets"
  ],
  "title": "DatasetPlan",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_plan/gen_plan_dataset_1/.sdk_openhands_agent_struct_out.json`.

Build and evaluate a simple MinHash near-duplicate detector for short text documents.
````

### [6] SKILL-INPUT — aii-json · 2026-06-22 05:05:36 UTC

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
