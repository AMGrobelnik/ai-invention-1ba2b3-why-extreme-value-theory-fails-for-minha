# gen_plan_experiment_2 — test_idea

> Phase: `invention_loop` · round 2 · `gen_plan`
> Run: `run_EqcgJR2naF4b` — Why Extreme Value Theory Fails for MinHash Confidence Intervals on Short Text (and What to Do Instead)
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_plan_experiment_2` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 05:46:02 UTC

````
<hypothesis>
kind: hypothesis
title: >-
  EVT-MinHash Negative Results: Why Extreme Value Theory Fails for MinHash Confidence Intervals with Short Text Documents
hypothesis: >-
  The Gumbel distribution (Extreme Value Type I) does NOT provide an adequate fit for MinHash signature minima from short
  text documents (10-100 shingles), due to (1) finite-sample bias where the number of shingles is insufficient for asymptotic
  convergence, (2) dependence between overlapping shingles violating the i.i.d. assumption, and (3) hash function discretization
  effects. Consequently, EVT-based confidence intervals are not suitable for MinHash-based near-duplicate detection in the
  small-sample regime. Practitioners should use alternative uncertainty quantification methods such as bootstrap (with corrected
  implementation), analytical binomial bounds based on the matching hash count distribution, or Bayesian approaches with informative
  priors on Jaccard similarity.
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
_relation_rationale: >-
  Refining from positive EVT claim to negative result based on strong experimental evidence (KS p-values < 1e-20)
_confidence_delta: decreased
_key_changes:
- >-
  Reversed core claim: EVT does NOT work for MinHash with short text (experiment showed KS p-values < 0.05)
- >-
  Added specific failure reasons: finite-sample bias, dependence violation, hash discretization
- >-
  Added actionable guidance: practitioners should use alternative methods (bootstrap, binomial bounds, Bayesian)
- >-
  Narrowed scope: explicitly limited to short text documents (10-100 shingles) where asymptotic EVT fails
- >-
  Removed unsubstantiated claims about computational efficiency of EVT (moot if method doesn't work)
- Added requirement to fix bootstrap implementation as baseline comparison
relation_type: evolution
</hypothesis>

<artifact_direction>
Make this direction concrete and actionable. Keep the same type and respect dependencies.

id: experiment_iter2_dir2
type: experiment
objective: >-
  Implement alternative uncertainty quantification methods for MinHash as practical alternatives to EVT: analytical binomial
  confidence intervals and Bayesian approach with informative priors
approach: >-
  1. ANALYTICAL BINOMIAL BOUNDS: Derive confidence intervals based on the binomial distribution of matching hash counts. For
  k hash functions, the number of matches follows Binomial(k, J). Use Clopper-Pearson exact CI or Wilson score interval to
  get CI for J from the observed match count. This is O(1) computation after MinHash. 2. BAYESIAN APPROACH: Place Beta prior
  on Jaccard similarity (Beta(α,β) with parameters informed by document length). Update with binomial likelihood from match
  count to get posterior distribution. Compute credible intervals from posterior. 3. Implement both methods and verify their
  coverage properties via simulation. 4. Compare computational cost: analytical binomial O(1), Bayesian O(computation of posterior),
  vs bootstrap O(B×k). 5. Test on real short text data from art_qBzTyA1I1xtt. Output: implementation of both methods, coverage
  evaluation, computational cost analysis.
depends_on:
- id: art_qBzTyA1I1xtt
  label: dataset
  relation_type:
  relation_rationale:
- id: art_T61Vq8hqL41x
  label: validates
  relation_type:
  relation_rationale:
</artifact_direction>

<dependencies>
Completed artifacts this artifact can use during execution.

--- Dependency 1 ---
id: art_T61Vq8hqL41x
type: research
title: 'EVT-MinHash: Extreme Value Theory for MinHash Confidence Intervals'
summary: >-
  Comprehensive theoretical research establishing the connection between MinHash and Extreme Value Theory (EVT). The Fisher-Tippett-Gnedenko
  theorem justifies modeling MinHash signatures as Gumbel-distributed random variables. Key findings: (1) The minimum of n
  i.i.d. uniform(0,1) variables (MinHash) converges to exponential distribution as n→∞, which transforms to Gumbel; (2) Gumbel
  parameters can be estimated via MLE or method of moments; (3) The delta method can propagate uncertainty to Jaccard similarity;
  (4) EVT-MinHash costs O(k) vs O(B×k) for bootstrap; (5) Related work (LSHBloom, FracMinHash, SetSketch) lacks statistically
  principled uncertainty quantification. The research includes complete mathematical derivations, assumption verification,
  limitation analysis, and a detailed related work comparison. The main open challenge is deriving the exact function J =
  g(μ_A, σ_A, μ_B, σ_B) relating Gumbel parameters to Jaccard similarity.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_research_1
out_expected_files:
- research_out.json
out_dependency_files:
  file_list:
  - research_out.json

--- Dependency 2 ---
id: art_qBzTyA1I1xtt
type: dataset
title: Short Text Datasets for EVT-MinHash Near-Duplicate Detection Evaluation
summary: >-
  Collected and standardized 3 short text datasets from HuggingFace Hub for evaluating EVT-MinHash near-duplicate detection.
  Datasets include: (1) cardiffnlp/tweet_eval (sentiment) with 10,000 tweets (avg 19.2 words), (2) cardiffnlp/tweet_eval (emoji)
  with 10,000 tweets (avg 11.8 words), and (3) fancyzhx/ag_news with 19,951 news headlines (avg 38.7 words). All datasets
  contain documents with 10-100 words meeting the artifact criteria. Total documents: 39,951. Total size: 17.0 MB (well under
  300MB limit). Each document includes doc_id, text, source, word count, and k=3 shingle count for MinHash evaluation. Datasets
  were validated for provenance (39K+ downloads for tweet_eval, 118K+ for ag_news), structure, and text quality. Output files
  include full_data_out.json (schema-validated), mini_data_out.json (3 examples per dataset), and preview_data_out.json (truncated
  text for quick inspection). The data.py script loads from temp/datasets/, standardizes to exp_sel_data_out.json schema,
  and saves to full_data_out.json.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1
out_expected_files:
- data.py
- full_data_out.json
- preview_data_out.json
- mini_data_out.json
out_dependency_files:
  file_list:
  - data.py
  - full_data_out.json
  - mini_data_out.json
  - preview_data_out.json
  data_file_paths:
  - full_data_out.json
  - mini_data_out.json
  - preview_data_out.json
</dependencies>

<instructions>
YOUR ROLE: Write a detailed PLAN for the artifact. A separate executor agent runs the actual artifact later.

You are a PLANNER, not an executor. Your output is a plan that tells the executor what to do and how.
Do NOT execute the artifact itself — a separate agent handles that. Your job is to plan it so well that the executor can follow your plan step by step.

You CAN and SHOULD: search the web, read papers, and explore library docs to make your plan concrete.
You CANNOT run shell commands or scripts — code execution is disabled. Research via web tools only.

Do NOT do the executor's job: don't download datasets, don't implement code, don't run experiments, don't write proofs, don't compute evaluations.

<artifact_executor_scope>
IMPORTANT: Each artifact executor has a focused prompt that guides it to do ONE thing well. It will NOT perform tasks outside its scope — assigning the wrong work to the wrong artifact type wastes an iteration. Match the task to the right executor.

EXPERIMENT executor scope:
  Output: method_out.json with results (metrics, predictions, analysis) — the core computational work
  DOES: Implement and run methods/algorithms, compute metrics, compare approaches, produce quantitative results
  DOES NOT: Collect new datasets (depends on DATASET artifacts for input data), write formal proofs
  This is the right artifact for any code that processes data and produces results
</artifact_executor_scope>

<artifact_planning_rules>
EXPERIMENT: Must depend on at least one DATASET. Define clear metrics and baselines before running. Consider trying multiple method variations rather than a single approach.
</artifact_planning_rules>

<compute_profiles>
Choose the compute profile this artifact needs for execution.
Available profiles for experiment artifacts:
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

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_plan/gen_plan_experiment_2/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "description": "Plan for an EXPERIMENT artifact.",
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
    "implementation_pseudocode": {
      "description": "High-level pseudocode for the experiment implementation",
      "title": "Implementation Pseudocode",
      "type": "string"
    },
    "fallback_plan": {
      "description": "What to do if the primary approach fails - alternative methods, simplified versions",
      "title": "Fallback Plan",
      "type": "string"
    },
    "testing_plan": {
      "description": "How to validate the experiment works: start with small/fast tests, look for confirmation signals before running full-scale experiments",
      "title": "Testing Plan",
      "type": "string"
    }
  },
  "required": [
    "title",
    "implementation_pseudocode",
    "fallback_plan",
    "testing_plan"
  ],
  "title": "ExperimentPlan",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_plan/gen_plan_experiment_2/.sdk_openhands_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-06-22 05:46:02 UTC

```
Build and evaluate a simple MinHash near-duplicate detector for short text documents.
```

### [3] SKILL-INPUT — aii-web-tools · 2026-06-22 05:46:12 UTC

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

### [4] SYSTEM-USER prompt · 2026-06-22 05:59:08 UTC

````
YOUR PREVIOUS SESSION WAS INTERRUPTED: A single operation exceeded the 720s message timeout. Each individual operation must complete within 720s. Do NOT mock, skip, or compromise your execution — still do the real work. Try to make operations run faster if possible. If a command genuinely takes longer than 720s, split it into sequential parts that each complete within the time limit.

Last messages before interruption:
  - [agent_tool_call] invoke_skill: Tool: invoke_skill
{
  "name": "aii-web-tools"
}
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "Here's the result of running `cat -n` on /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/mini_data_out.json:\n     1\t{\n     2\t  \"metadata\": {\n     3\t    \"tota
  - [agent_tool_result] invoke_skill: Tool: invoke_skill
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "## Web tools\n\nYou have three web capabilities: **search**, **fetch**, and **grep** (exact\nregex extraction over a full page or PDF).\n\n**Pick where they come from, in this order:**\n\n1. **If you have built-
  - [agent_think]: The user wants me to create a detailed plan for an experiment that implements alternative uncertainty quantification methods for MinHash - specifically analytical binomial confidence intervals and Bayesian approaches.

Let me understand what I've learned so far:
1. The data structure from mini_data_
  - [agent_response]: Let me research the specific statistical methods needed for this plan - Clopper-Pearson, Wilson score interval, and Beta-binomial Bayesian approach.
  - [agent_tool_call] invoke_skill: Tool: invoke_skill
{
  "name": "aii-web-tools"
}
  - [agent_tool_result] invoke_skill: Tool: invoke_skill
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "## Web tools\n\nYou have three web capabilities: **search**, **fetch**, and **grep** (exact\nregex extraction over a full page or PDF).\n\n**Pick where they come from, in this order:**\n\n1. **If you have built-
  - [agent_think]: Now I need to plan this experiment carefully. Let me think about what the executor needs:

1. **Understanding the problem**: The hypothesis states that EVT fails for MinHash with short text, so we need practical alternatives.

2. **Two main methods to implement**:
   - Analytical binomial confidence
  - [agent_response]: Let me research the statistical methods needed and then create a comprehensive plan. I'll search for the exact formulas needed.
  - [agent_tool_call] think: Tool: think
{
  "thought": "Let me think about what I need to include in this plan:\n\n1. **Background understanding from dependencies**:\n   - Dataset has short text (10-100 words, 76-232 shingles)\n   - Research shows EVT fails for MinHash with short text\n   - Need to implement TWO alternative me
  - [agent_tool_result] think: Tool: think
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "Your thought has been logged."}], "is_error": false, "kind": "ThinkObservation"}
  - [agent_think]: Let me now create a comprehensive plan for the experiment. I need to:

1. Research the exact formulas for Clopper-Pearson, Wilson score interval, and Beta-binomial Bayesian approach
2. Create detailed pseudocode for the executor
3. Specify the evaluation methodology
4. Provide a fallback plan
5. Cre
  - [agent_response]: Now I have a good understanding of the dependencies and context. Let me create a detailed, actionable plan for implementing the alternative uncertainty quantification methods for MinHash. I'll include specific statistical formulas, implementation steps, and evaluation methodology.
  - [agent_tool_call] file_editor: Tool: file_editor
{
  "command": "create",
  "path": "/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_plan/gen_plan_experiment_2/experiment_plan.json",
  "file_text": "{\n  \"title\": \"Alternative Uncertainty Quantification for MinHash: Binomial and Bayesian Methods\",\n  \"
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "File created successfully at: /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_plan/gen_plan_experiment_2/experiment_plan.json"}], "is_error": false, "command": "create", "path": "/ai-inven
  - [agent_think]: Wait, I created the plan as a JSON file, but the instructions say I need to output the result as JSON to a specific file: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_plan/gen_plan_experiment_2/.sdk_openhands_agent_struct_out.json`

Let me check the required schema again:
  - [agent_response]: I've created a detailed plan, but I need to save it to the correct output file. Let me create the required output file in the correct location.
  - [agent_tool_call] file_editor: Tool: file_editor
{
  "command": "create",
  "path": "/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_plan/gen_plan_experiment_2/.sdk_openhands_agent_struct_out.json",
  "file_text": "{\n  \"title\": \"Alternative Uncertainty Quantification for MinHash: Binomial and Bayesian 
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "File created successfully at: /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_plan/gen_plan_experiment_2/.sdk_openhands_agent_struct_out.json"}], "is_error": false, "command": "create", "p
  - [status_public_warning]: [PauseEvent]

CONTINUE FOLLOWING THESE INSTRUCTIONS:

<hypothesis>
kind: hypothesis
title: >-
  EVT-MinHash Negative Results: Why Extreme Value Theory Fails for MinHash Confidence Intervals with Short Text Documents
hypothesis: >-
  The Gumbel distribution (Extreme Value Type I) does NOT provide an adequate fit for MinHash signature minima from short
  text documents (10-100 shingles), due to (1) finite-sample bias where the number of shingles is insufficient for asymptotic
  convergence, (2) dependence between overlapping shingles violating the i.i.d. assumption, and (3) hash function discretization
  effects. Consequently, EVT-based confidence intervals are not suitable for MinHash-based near-duplicate detection in the
  small-sample regime. Practitioners should use alternative uncertainty quantification methods such as bootstrap (with corrected
  implementation), analytical binomial bounds based on the matching hash count distribution, or Bayesian approaches with informative
  priors on Jaccard similarity.
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
_relation_rationale: >-
  Refining from positive EVT claim to negative result based on strong experimental evidence (KS p-values < 1e-20)
_confidence_delta: decreased
_key_changes:
- >-
  Reversed core claim: EVT does NOT work for MinHash with short text (experiment showed KS p-values < 0.05)
- >-
  Added specific failure reasons: finite-sample bias, dependence violation, hash discretization
- >-
  Added actionable guidance: practitioners should use alternative methods (bootstrap, binomial bounds, Bayesian)
- >-
  Narrowed scope: explicitly limited to short text documents (10-100 shingles) where asymptotic EVT fails
- >-
  Removed unsubstantiated claims about computational efficiency of EVT (moot if method doesn't work)
- Added requirement to fix bootstrap implementation as baseline comparison
relation_type: evolution
</hypothesis>

<artifact_direction>
Make this direction concrete and actionable. Keep the same type and respect dependencies.

id: experiment_iter2_dir2
type: experiment
objective: >-
  Implement alternative uncertainty quantification methods for MinHash as practical alternatives to EVT: analytical binomial
  confidence intervals and Bayesian approach with informative priors
approach: >-
  1. ANALYTICAL BINOMIAL BOUNDS: Derive confidence intervals based on the binomial distribution of matching hash counts. For
  k hash functions, the number of matches follows Binomial(k, J). Use Clopper-Pearson exact CI or Wilson score interval to
  get CI for J from the observed match count. This is O(1) computation after MinHash. 2. BAYESIAN APPROACH: Place Beta prior
  on Jaccard similarity (Beta(α,β) with parameters informed by document length). Update with binomial likelihood from match
  count to get posterior distribution. Compute credible intervals from posterior. 3. Implement both methods and verify their
  coverage properties via simulation. 4. Compare computational cost: analytical binomial O(1), Bayesian O(computation of posterior),
  vs bootstrap O(B×k). 5. Test on real short text data from art_qBzTyA1I1xtt. Output: implementation of both methods, coverage
  evaluation, computational cost analysis.
depends_on:
- id: art_qBzTyA1I1xtt
  label: dataset
  relation_type:
  relation_rationale:
- id: art_T61Vq8hqL41x
  label: validates
  relation_type:
  relation_rationale:
</artifact_direction>

<dependencies>
Completed artifacts this artifact can use during execution.

--- Dependency 1 ---
id: art_T61Vq8hqL41x
type: research
title: 'EVT-MinHash: Extreme Value Theory for MinHash Confidence Intervals'
summary: >-
  Comprehensive theoretical research establishing the connection between MinHash and Extreme Value Theory (EVT). The Fisher-Tippett-Gnedenko
  theorem justifies modeling MinHash signatures as Gumbel-distributed random variables. Key findings: (1) The minimum of n
  i.i.d. uniform(0,1) variables (MinHash) converges to exponential distribution as n→∞, which transforms to Gumbel; (2) Gumbel
  parameters can be estimated via MLE or method of moments; (3) The delta method can propagate uncertainty to Jaccard similarity;
  (4) EVT-MinHash costs O(k) vs O(B×k) for bootstrap; (5) Related work (LSHBloom, FracMinHash, SetSketch) lacks statistically
  principled uncertainty quantification. The research includes complete mathematical derivations, assumption verification,
  limitation analysis, and a detailed related work comparison. The main open challenge is deriving the exact function J =
  g(μ_A, σ_A, μ_B, σ_B) relating Gumbel parameters to Jaccard similarity.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_research_1
out_expected_files:
- research_out.json
out_dependency_files:
  file_list:
  - research_out.json

--- Dependency 2 ---
id: art_qBzTyA1I1xtt
type: dataset
title: Short Text Datasets for EVT-MinHash Near-Duplicate Detection Evaluation
summary: >-
  Collected and standardized 3 short text datasets from HuggingFace Hub for evaluating EVT-MinHash near-duplicate detection.
  Datasets include: (1) cardiffnlp/tweet_eval (sentiment) with 10,000 tweets (avg 19.2 words), (2) cardiffnlp/tweet_eval (emoji)
  with 10,000 tweets (avg 11.8 words), and (3) fancyzhx/ag_news with 19,951 news headlines (avg 38.7 words). All datasets
  contain documents with 10-100 words meeting the artifact criteria. Total documents: 39,951. Total size: 17.0 MB (well under
  300MB limit). Each document includes doc_id, text, source, word count, and k=3 shingle count for MinHash evaluation. Datasets
  were validated for provenance (39K+ downloads for tweet_eval, 118K+ for ag_news), structure, and text quality. Output files
  include full_data_out.json (schema-validated), mini_data_out.json (3 examples per dataset), and preview_data_out.json (truncated
  text for quick inspection). The data.py script loads from temp/datasets/, standardizes to exp_sel_data_out.json schema,
  and saves to full_data_out.json.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_dataset_1
out_expected_files:
- data.py
- full_data_out.json
- preview_data_out.json
- mini_data_out.json
out_dependency_files:
  file_list:
  - data.py
  - full_data_out.json
  - mini_data_out.json
  - preview_data_out.json
  data_file_paths:
  - full_data_out.json
  - mini_data_out.json
  - preview_data_out.json
</dependencies>

<instructions>
YOUR ROLE: Write a detailed PLAN for the artifact. A separate executor agent runs the actual artifact later.

You are a PLANNER, not an executor. Your output is a plan that tells the executor what to do and how.
Do NOT execute the artifact itself — a separate agent handles that. Your job is to plan it so well that the executor can follow your plan step by step.

You CAN and SHOULD: search the web, read papers, and explore library docs to make your plan concrete.
You CANNOT run shell commands or scripts — code execution is disabled. Research via web tools only.

Do NOT do the executor's job: don't download datasets, don't implement code, don't run experiments, don't write proofs, don't compute evaluations.

<artifact_executor_scope>
IMPORTANT: Each artifact executor has a focused prompt that guides it to do ONE thing well. It will NOT perform tasks outside its scope — assigning the wrong work to the wrong artifact type wastes an iteration. Match the task to the right executor.

EXPERIMENT executor scope:
  Output: method_out.json with results (metrics, predictions, analysis) — the core computational work
  DOES: Implement and run methods/algorithms, compute metrics, compare approaches, produce quantitative results
  DOES NOT: Collect new datasets (depends on DATASET artifacts for input data), write formal proofs
  This is the right artifact for any code that processes data and produces results
</artifact_executor_scope>

<artifact_planning_rules>
EXPERIMENT: Must depend on at least one DATASET. Define clear metrics and baselines before running. Consider trying multiple method variations rather than a single approach.
</artifact_planning_rules>

<compute_profiles>
Choose the compute profile this artifact needs for execution.
Available profiles for experiment artifacts:
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

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_plan/gen_plan_experiment_2/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "description": "Plan for an EXPERIMENT artifact.",
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
    "implementation_pseudocode": {
      "description": "High-level pseudocode for the experiment implementation",
      "title": "Implementation Pseudocode",
      "type": "string"
    },
    "fallback_plan": {
      "description": "What to do if the primary approach fails - alternative methods, simplified versions",
      "title": "Fallback Plan",
      "type": "string"
    },
    "testing_plan": {
      "description": "How to validate the experiment works: start with small/fast tests, look for confirmation signals before running full-scale experiments",
      "title": "Testing Plan",
      "type": "string"
    }
  },
  "required": [
    "title",
    "implementation_pseudocode",
    "fallback_plan",
    "testing_plan"
  ],
  "title": "ExperimentPlan",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_plan/gen_plan_experiment_2/.sdk_openhands_agent_struct_out.json`.

Build and evaluate a simple MinHash near-duplicate detector for short text documents.
````
