# gen_full_paper — report_results

> Phase: `gen_paper_repo` · `gen_full_paper`
> Run: `run_EqcgJR2naF4b` — Why Extreme Value Theory Fails for MinHash Confidence Intervals on Short Text (and What to Do Instead)
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_full_paper` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 07:53:52 UTC

````
<workspace>
Your workspace: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/`:
GOOD: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/file.py`, `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>

<task>
Create a publication-ready top-conference LaTeX paper with BibTeX from <paper_text> and <available_figures>, compile to PDF.
</task>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<paper_text>
title: >-
  Why Extreme Value Theory Fails for MinHash Confidence Intervals on Short Text (and What to Do Instead)
abstract: >-
  MinHash is a widely used algorithm for estimating Jaccard similarity in near-duplicate detection. However, quantifying uncertainty
  in MinHash estimates remains challenging, especially for short text documents where the small number of shingles leads to
  high-variance estimates. We investigate whether Extreme Value Theory (EVT) can provide statistically principled confidence
  intervals for MinHash-based Jaccard estimates. Through theoretical analysis and comprehensive evaluation on 3,000 document
  pairs from real-world datasets (tweet_eval, ag_news), we show that the Gumbel and Weibull distributions—the theoretical
  limits for minima of i.i.d. random variables—provide poor fits to MinHash signature minima from short text documents (10-100
  shingles), with Kolmogorov-Smirnov test p-values < 10^{-20}. We identify three causes: finite-sample bias, dependence between
  overlapping shingles, and hash function discretization. To address this, we implement and evaluate five uncertainty quantification
  methods: EVT-Gumbel, EVT-Weibull, Corrected Bootstrap, Analytical Binomial (Clopper-Pearson), and Bayesian (Beta prior).
  Our evaluation on 3,000 document pairs shows that Analytical Binomial and Bayesian methods achieve 96.5% and 94.8% coverage
  (within 2% of the nominal 95% level), compared to 75.5% for Corrected Bootstrap with 1,000 resamples. Computation times
  are 84μs for Clopper-Pearson, 41μs for Wilson score, 395μs for Bayesian, and 2,000+μs for Bootstrap per confidence interval.
  We provide practitioner guidelines and open-source implementations for uncertainty quantification in MinHash-based near-duplicate
  detection.
paper_text: |-
  # Introduction

  Near-duplicate detection is a fundamental task in information retrieval, content moderation, and data cleaning. The MinHash algorithm [1] estimates Jaccard similarity between sets by comparing minimum hash values of their elements, providing a computationally efficient approach for large-scale deduplication. However, MinHash is fundamentally a statistical estimator, and its uncertainty quantification remains an open challenge, particularly for short text documents where the number of shingles (contiguous subsequences) is small.

  For short text documents such as tweets, SMS messages, and news headlines, the small number of shingles (typically 10-100) leads to high-variance Jaccard similarity estimates. Practitioners currently rely on heuristics or computationally expensive bootstrap methods to assess confidence in MinHash estimates. Bootstrap methods require resampling the data many times (typically 1,000-10,000 resamples) to construct confidence intervals, resulting in computational costs that scale as O(B*k) where B is the number of bootstrap samples and k is the number of hash functions.

  This paper explores whether Extreme Value Theory (EVT) can provide a theoretically grounded and computationally efficient alternative to bootstrap for uncertainty quantification in MinHash-based Jaccard similarity estimation. The core hypothesis is that MinHash signature values—being the minimum of uniform random variables—follow a Gumbel distribution (Extreme Value Type I) under the Fisher-Tippett-Gnedenko theorem [2]. If this hypothesis holds, it would enable closed-form confidence intervals and hypothesis tests for Jaccard similarity estimates with computational cost of only O(k) operations.

  ## Problem Statement

  Given two sets A and B, the Jaccard similarity is defined as:

  \[J(A, B) = \frac{|A \cap B|}{|A \cup B|}\]

  MinHash estimates Jaccard similarity by computing k independent hash functions and keeping the minimum hash value for each function. The probability that MinHash values match equals the Jaccard similarity [1]. However, for finite k and small set sizes, the MinHash estimate has variance that depends on the underlying distribution of hash values.

  The key research question is: *Does the distribution of MinHash signature minima follow a Gumbel distribution for short text documents, and if so, can we derive statistically principled confidence intervals for Jaccard similarity estimates?*

  ## Contributions

  This paper makes the following contributions:

  1. **Theoretical Analysis**: We derive the theoretical connection between MinHash and Extreme Value Theory, showing that the minimum of n i.i.d. uniform(0,1) variables converges to an exponential distribution as n→∞, which connects to the Gumbel distribution. We also identify the key assumptions (i.i.d., asymptotic regime) and their violations in practice.

  2. **Empirical Verification**: We implement MinHash with 128 hash functions using MD5 and empirically test whether MinHash signature minima follow Gumbel or Weibull distributions for short text documents (10-100 character shingles). We use real-world datasets from HuggingFace Hub (tweet_eval, ag_news) with 39,951 documents \footnote{Code: \url{https://github.com/AMGrobelnik/ai-invention-1ba2b3-why-extreme-value-theory-fails-for-minha/tree/main/round-1/dataset-1}}.

  3. **Distribution Fit Analysis**: We perform Kolmogorov-Smirnov (KS) tests and Anderson-Darling (AD) tests to evaluate Gumbel and Weibull distribution fits across three datasets and multiple shingle count ranges (10-30, 30-50, 50-100).

  4. **Negative Results on EVT**: We report that neither Gumbel nor Weibull distributions provide adequate fit for MinHash minima from short text documents, with all KS test p-values < 10^{-20}. This suggests that the asymptotic EVT approximation does not hold in the small-sample regime due to finite-sample bias, dependence between overlapping shingles, and hash function discretization.

  5. **Alternative UQ Methods**: We implement and evaluate two practical alternatives to EVT-MinHash: (a) analytical binomial confidence intervals (Clopper-Pearson exact and Wilson score) based on matching hash counts, and (b) Bayesian approach with Beta prior informed by document length \footnote{Code: \url{https://github.com/AMGrobelnik/ai-invention-1ba2b3-why-extreme-value-theory-fails-for-minha/tree/main/round-2/experiment-2}}.

  6. **Comprehensive Evaluation**: We compare five uncertainty quantification methods (EVT-Gumbel, EVT-Weibull, Corrected Bootstrap, Analytical Binomial, Bayesian) on 3,000 document pairs from real-world datasets. Results show that Analytical Binomial and Bayesian methods achieve 96.5% and 94.8% coverage (within 2% of 95% target), while Corrected Bootstrap only achieves 75.5% coverage \footnote{Code: \url{https://github.com/AMGrobelnik/ai-invention-1ba2b3-why-extreme-value-theory-fails-for-minha/tree/main/round-2/evaluation-1}}.

  7. **Computational Cost Analysis**: We benchmark all methods and show that analytical approaches are 10-50x faster than bootstrap, with Clopper-Pearson taking 84μs, Wilson score 41μs, Bayesian 395μs, and Bootstrap 2,000+μs per confidence interval.

  # Related Work

  ## MinHash and LSH

  MinHash was introduced by Broder [1] for estimating resemblance and containment between documents. The algorithm computes the minimum hash value for each set and uses the fraction of matching minima to estimate Jaccard similarity. Subsequent work has extended MinHash to weighted sets [3], improved memory efficiency using Bloom filters (LSHBloom [4]), and developed sampling-based approaches for Jaccard estimation [5].

  ## Uncertainty Quantification for MinHash

  Existing work on uncertainty quantification for MinHash includes:

  1. **FracMinHash** [6]: Derives confidence intervals for FracMinHash in bioinformatics applications using asymptotic normality. However, the normal approximation may fail for small sketches or extreme similarities.

  2. **Sampling-Based Jaccard Estimation** [5]: Introduces a binomial model for predicting overlap between random samples of sets, providing error bounds and sample size requirements. This approach requires storing random samples and is not directly applicable to MinHash sketches.

  3. **Bootstrap Methods** [7]: Bootstrap confidence intervals are a common approach for uncertainty quantification but require many resamples (B = 1,000-10,000) for accurate coverage, resulting in high computational cost.

  ## Extreme Value Theory in Sketching

  SetSketch [8] uses the Gumbel distribution for cardinality estimation (counting distinct elements) but not for Jaccard similarity confidence intervals. To the best of our knowledge, this is the first work to investigate whether MinHash signature values follow an extreme value distribution for the purpose of constructing confidence intervals for Jaccard similarity, and to provide a comprehensive comparison of alternative uncertainty quantification methods.

  # Methods

  ## MinHash Implementation

  We implement MinHash with k = 128 independent hash functions using MD5. Each hash function is generated with a different seed. For a document with text t, we extract character shingles of length k_shingle = 3. The Jaccard similarity estimate is the fraction of matching signature values.

  ## Extreme Value Theory Background

  The Fisher-Tippett-Gnedenko theorem states that the minimum of n i.i.d. random variables converges to one of three extreme value distributions. For uniform(0,1) variables, the limiting distribution is Gumbel.

  ### Assumptions and Limitations of EVT for MinHash

  The application of EVT to MinHash relies on: (1) i.i.d. hash values (violated by overlapping shingles), (2) asymptotic regime (n→∞, not satisfied for 10-100 shingles), (3) uniform hash functions (approximated by MD5), (4) continuous distribution (violated by discrete MD5 outputs).

  ## Alternative Uncertainty Quantification Methods

  ### Analytical Binomial Confidence Intervals

  The number of matching hashes out of k follows a binomial distribution. We implement Clopper-Pearson (exact) and Wilson score intervals.

  ### Bayesian Approach

  We use a Beta prior on Jaccard similarity, updated with observed match count. We use uniform prior (α=1, β=1) and length-informed prior.

  ### Corrected Bootstrap

  The bootstrap method resamples shingles (not signatures) to preserve dependence. For each resample, we recompute signatures and Jaccard estimate.

  # Experiments

  ## Datasets

  We evaluate on real-world datasets from HuggingFace Hub: Tweet Eval Sentiment (10K tweets, avg 68.5 shingles), Tweet Eval Emoji (10K tweets, avg 53.2 shingles), AG News (19,951 headlines, avg 156.8 shingles). Total: 39,951 documents .

  [FIGURE:fig1]

  ## Experimental Setup

  We generate 3,000 document pairs, compute true and estimated Jaccard, fit EVT distributions, and evaluate five UQ methods.

  # Results

  ## Distribution Fit Results

  Table 1 shows KS test results. All p-values < 10^{-20}, indicating poor fit.

  **Table 1**: Distribution fit results. All KS p-values < 10^{-20}.

  [FIGURE:fig2]

  ## Comprehensive Evaluation of UQ Methods

  Table 2 shows coverage probability for 95% CIs.

  **Table 2**: Coverage probability. Analytical Binomial: 96.5%, Bayesian: 94.8%, Corrected Bootstrap: 75.5%.

  ## Confidence Interval Width

  **Table 3**: CI width. Analytical Binomial: mean=0.0996, Bayesian: mean=0.0935, Corrected Bootstrap: mean=0.0543.

  ## Bias and RMSE

  **Table 4**: Bias and RMSE. All methods have similar bias (~0.0026) except Bayesian (0.0091).

  ## Computational Cost

  **Table 5**: Computational cost. Clopper-Pearson: 84μs, Wilson: 41μs, Bayesian: 395μs, Bootstrap: 2,000+μs.

  [FIGURE:fig3]

  # Discussion

  ## Interpretation of Results

  EVT distributions provide poor fit due to: (1) small sample size, (2) dependence between shingles, (3) hash function bias, (4) discrete distribution. Despite this, EVT methods achieve acceptable coverage, suggesting robustness but lacking theoretical justification.

  ## Recommendations for Practitioners

  1. Use Analytical Binomial or Bayesian for well-calibrated CIs.
  2. Avoid EVT for short text (<100 shingles).
  3. Increase bootstrap resamples if using bootstrap.
  4. Method selection: Speed → Wilson Score (41μs), Rigor → Clopper-Pearson, Priors → Bayesian.

  # Conclusion

  We show that EVT distributions provide poor fits to MinHash minima from short text. Analytical Binomial and Bayesian methods achieve 96.5% and 94.8% coverage. Analytical methods are 10-50x faster than bootstrap. We provide practitioner guidelines and open-source implementations.

  # Acknowledgments

  This work was supported by the AI Inventor automated research system.

  # References

  [1] Broder, A. Z. (1997). On the resemblance and containment of documents.
  [2] Fisher & Tippett (1928). Limiting forms of the frequency distribution.
  [3] Wu et al. (2018). A review for weighted MinHash algorithms.
  [4] Khan et al. (2024). LSHBloom. arXiv:2411.04257.
  [5] Joshi (2025). Sampling-Based Estimation of Jaccard. arXiv:2507.10019.
  [6] Irber & Brown (2023). Debiasing FracMinHash. PLoS Comput Biol.
  [7] Efron & Tibshirani (1994). An introduction to the bootstrap.
  [8] Ertl (2021). SetSketch. VLDB.
summary: >-
  This paper investigates EVT for uncertainty quantification in MinHash-based Jaccard estimation. We show that EVT distributions
  provide poor fits to MinHash minima from short text (KS p-values < 10^{-20}). We implement and evaluate five UQ methods
  on 3,000 document pairs. Results show that Analytical Binomial and Bayesian methods achieve 96.5% and 94.8% coverage, while
  Corrected Bootstrap only achieves 75.5%. Analytical methods are 10-50x faster than bootstrap.
</paper_text>

<available_figures>
--- Item 1 ---
id: fig1
title: Dataset Shingle Count Distributions
caption: Distribution of shingle counts across three datasets.
image_gen_detailed_description: Three panel histogram. White background, sans-serif font.
aspect_ratio: '21:9'
summary: Shingle count distributions
figure_path: figures/fig1_v0.jpg

--- Item 2 ---
id: fig2
title: 'QQ Plots: EVT Distribution Fit'
caption: QQ plots showing poor fit of EVT distributions to MinHash minima.
image_gen_detailed_description: Two panel QQ plot. White background, sans-serif font.
aspect_ratio: '21:9'
summary: QQ plots showing poor EVT fit
figure_path: figures/fig2_v0.jpg

--- Item 3 ---
id: fig3
title: Computational Cost Comparison
caption: Computational cost per CI for five UQ methods.
image_gen_detailed_description: Grouped bar chart, log scale. White background, sans-serif font.
aspect_ratio: '21:9'
summary: Computational cost comparison
figure_path: figures/fig3_v0.jpg
</available_figures>

<figure_requirements>
CRITICAL: Include ALL figures from <available_figures>. No exceptions.

- Every figure MUST use \includegraphics{figures/filename.jpg}
- Do NOT skip, convert to tables, or describe without inserting
- Each needs: \begin{figure*|figure}[placement], \includegraphics, \caption, \label, \end{...} — pick env + placement by the figure's `aspect_ratio` field (see PLACEMENT below). Constrain every \includegraphics with `width=\linewidth,height=0.4\textheight,keepaspectratio` (single-column) or `width=\textwidth,height=0.45\textheight,keepaspectratio` (figure*). Use exactly these option keys — `max height=` is NOT valid LaTeX
- Use the `caption` field from each figure for \caption{...} — do NOT invent new captions
- Place figures where their [FIGURE:fig_id] markers appear in paper_text
- VERIFICATION: paper.tex MUST have exact same number of \includegraphics as <available_figures>
- Do NOT generate new figure images (no matplotlib, no PIL, no image generation). Use ONLY the pre-generated figures from <available_figures>. They were already created by a previous pipeline step.

PLACEMENT BY ASPECT RATIO (use the `aspect_ratio` field on each figure):
- `21:9` (architecture diagrams / hero figures): \begin{figure*}[!t] (full two-column width, top of page). The hero architecture diagram should appear EARLY in the paper — typically at the top of page 2. Marker placement in paper_text already determines this; preserve it.
- `16:9` (comparisons, multi-panel results): \begin{figure*}[!t] for full-width or \begin{figure}[!htbp] for single-column.
- `4:3` / `1:1` / `3:2` / `3:4` / `9:16`: \begin{figure}[!htbp] (single-column).
</figure_requirements>

<artifact_links>
The paper_text contains \footnote{Code: \url{...}} references linking to artifact source code
on GitHub. Include \usepackage{hyperref} and \usepackage{url}.
Preserve these exactly as-is — do not remove, rewrite, or convert them to plain text.
The URLs will not resolve yet (the repo is deployed after compilation) — do NOT try to verify or fix them.
</artifact_links>

<headings>
NEVER use inline math (``$...$``) inside ``\section{...}`` / ``\subsection{...}`` / ``\subsubsection{...}`` arguments — hyperref's bookmark builder errors out (``Token not allowed in a PDF string``) and the PDF outline breaks. If a section heading needs a math-looking term, use the text equivalent (``d star`` not ``$d^*$``, ``alpha-equivalent`` not ``$\alpha$-equivalent``) or wrap it in ``\texorpdfstring{$math$}{plain}``. Inline math inside body paragraphs is fine.
</headings>

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Read and STRICTLY follow these skills: aii-paper-to-latex, aii-semscholar-bib.
TODO 2. Review <paper_text> and <available_figures>. Copy all figure images into ./figures/ in your workspace. Count figures — MUST include every one. Plan placements per section. Build `./references.bib` via aii_semscholar_bib__fetch — collect DOIs/ArXiv IDs from <paper_text> and batch-fetch all BibTeX in one call. Do NOT fabricate entries.
TODO 3. Create `./paper.tex` per aii-paper-to-latex skill's setup, write ALL sections, insert ALL figures from <available_figures>, include `./references.bib` via \bibliography. Compile to PDF per skill's process. Fix errors.
TODO 4. CRITICAL VERIFICATION: Run `grep -c 'includegraphics' paper.tex`, confirm count equals figures in <available_figures>. If not, add missing figures. Verify `./paper.pdf` was created.
TODO 5. VISUAL REVIEW: Write Python script to convert EVERY page of paper.pdf to PNG at 150 DPI (use pdf2image or pymupdf). Then read ALL page screenshots — each page image costs ~1,600 tokens so a 15-page paper is only ~24K tokens. You MUST read every page. The ONLY exception is if all page images would not fit in your remaining context — in that case, read as many as fit and state which pages you are skipping and why. Check every page for layout issues, overlapping figures, cut-off text, bad spacing, formatting problems. Fix issues and recompile.
TODO 6. FINAL READ: Check page count (`pdfinfo paper.pdf` or pymupdf). Read entire paper.pdf — check for missing sections, unclear explanations, inconsistencies, typos. Fix and recompile. The ONLY exception is if all pages would not fit in your remaining context — in that case, read as many pages as fit and state which pages you are skipping and why.
</todos>

---

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "FullPaperExpectedFiles": {
      "description": "All expected output files from full paper generation.",
      "properties": {
        "paper_tex_path": {
          "description": "Path to LaTeX source file. Example: 'paper.tex'",
          "title": "Paper Tex Path",
          "type": "string"
        },
        "paper_pdf_path": {
          "description": "Path to compiled PDF. Example: 'paper.pdf'",
          "title": "Paper Pdf Path",
          "type": "string"
        },
        "references_bib_path": {
          "description": "Path to BibTeX bibliography file. Example: 'references.bib'",
          "title": "References Bib Path",
          "type": "string"
        },
        "figure_paths": {
          "description": "Paths to all figure image files. Example: ['figures/fig1_v0.jpg', 'figures/fig2_v0.jpg']",
          "items": {
            "type": "string"
          },
          "title": "Figure Paths",
          "type": "array"
        }
      },
      "required": [
        "paper_tex_path",
        "paper_pdf_path",
        "references_bib_path",
        "figure_paths"
      ],
      "title": "FullPaperExpectedFiles",
      "type": "object"
    }
  },
  "description": "Full paper \u2014 structured output from paper generation.",
  "properties": {
    "title": {
      "description": "Short descriptive title for this paper generation task (roughly 30-90 characters)",
      "maxLength": 90,
      "minLength": 30,
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "description": "Brief summary of the generated paper: sections written, figures included, compilation status",
      "maxLength": 5000,
      "minLength": 500,
      "title": "Summary",
      "type": "string"
    },
    "out_expected_files": {
      "$ref": "#/$defs/FullPaperExpectedFiles",
      "description": "All output files you created. Must include paper.tex, paper.pdf, references.bib, and paths to all figure files."
    }
  },
  "required": [
    "title",
    "summary",
    "out_expected_files"
  ],
  "title": "FullPaper",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/.sdk_openhands_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-06-22 07:53:52 UTC

```
Build and evaluate a simple MinHash near-duplicate detector for short text documents.
```

### [3] SKILL-INPUT — aii-paper-to-latex · 2026-06-22 07:54:12 UTC

The agent loaded the **aii-paper-to-latex** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-paper-to-latex
description: LaTeX paper assembly and compilation. Covers document setup, figure inclusion from pre-generated JPEGs, compilation process, and output files. Use when assembling a paper from pre-written text and pre-generated figures into a compiled PDF.
---

## LaTeX Paper Assembly

Assembles a research paper from paper text, pre-generated figure JPEGs, and bibliography into a compiled PDF.

### Document Setup

```latex
\documentclass[11pt,letterpaper]{article}
\usepackage{graphicx, geometry, amsmath, hyperref, natbib, booktabs, xcolor, listings}
\geometry{margin=1in}
\hypersetup{colorlinks=true, linkcolor=black, citecolor=black, urlcolor=black}
```

### Figure Inclusion

CRITICAL: Include ALL figures. Every figure MUST appear in the paper.

```latex
\begin{figure}[!htbp]
  \centering
  \includegraphics[width=0.92\textwidth,keepaspectratio]{figures/filename.jpg}
  \caption{Descriptive caption.}
  \label{fig:label}
\end{figure}
```

Rules:
- ALWAYS use `[!htbp]` float placement (NOT `[t]` or `[h]` alone)
- ALWAYS constrain with `width` and `keepaspectratio` to prevent page takeover
- Every figure needs `\caption`, `\label`, and a `\ref` in the text
- Do NOT convert figures to tables or describe them without inserting the image
- Do NOT skip any figures

### Compilation Process

Run each command separately (do NOT chain with `&&` — pdflatex often exits non-zero on warnings, which would skip bibtex and leave citations as `??`):

```bash
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

All four commands are required. Skipping bibtex causes `??` in all citations.
Fix any errors between runs. Verify `./paper.pdf` was created.

### Output Files

- `./paper.tex` — LaTeX source
- `./references.bib` — bibliography file
- `./paper.pdf` — compiled PDF
- `./figures/*.jpg` — all figure images (pre-generated, copied into workspace)
````

### [4] SKILL-INPUT — aii-semscholar-bib · 2026-06-22 07:54:12 UTC

The agent loaded the **aii-semscholar-bib** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-semscholar-bib
description: Build bibliographies using Semantic Scholar. Batch-fetch BibTeX for papers by DOI, ArXiv ID, or title. Use when writing papers, generating reference lists, or building .bib files.
---

## Tool: `aii_semscholar_bib__fetch`

Batch-fetch BibTeX entries from Semantic Scholar. Pass all references in a single call — the tool handles batching internally.

### How it works

1. **DOI/ArXiv refs** → batched into POST /paper/batch calls (up to 500 per API call, auto-chunked)
2. **Title-only refs** → individual GET /paper/search/match (1s delay between)
3. **Post-process** → fix entry type, fix citation key (AuthorYYYY), inject DOI

The ability server runs a single worker (`max_threads: 1`). Multiple concurrent tool calls are queued — each runs independently (no cross-request aggregation). Batching happens within each request.

### Input format

```json
{
  "references": [
    {"doi": "10.48550/arXiv.1706.03762", "author": "Vaswani", "year": 2017},
    {"arxiv": "2201.11903", "author": "Wei", "year": 2022},
    {"title": "Tree of Thoughts", "author": "Yao", "year": 2023}
  ]
}
```

Each reference object can have:
- `doi` — DOI string (ArXiv DOIs like `10.48550/arXiv.XXXX.XXXXX` auto-convert to ArXiv IDs)
- `arxiv` — ArXiv ID (e.g. `"2305.14325"`)
- `title` — Paper title (used for search/match when no DOI/ArXiv)
- `author` — First author last name (for cleaner citation key)
- `year` — Publication year (int, for citation key)

At least one of `doi`, `arxiv`, or `title` is required per reference.

### Output format

```json
{
  "success": true,
  "bib_text": "@inproceedings{Vaswani2017, ...}\n\n@article{Wei2022, ...}",
  "total": 3,
  "found": 3,
  "failed_count": 0,
  "entries": [{"citation_key": "Vaswani2017", "bibtex": "...", "title": "...", "doi": "...", "arxiv": ""}],
  "failed": []
}
```

### Workflow

1. Collect DOIs, ArXiv IDs, or titles for all papers you need to cite
2. Call `aii_semscholar_bib__fetch` with the full list in **one call**
3. Save `bib_text` from the response to your `references.bib` file
4. Check `failed` — for any missed papers, follow the **fallback procedure** below

### Fallback for failed references (MANDATORY)

NEVER fabricate BibTeX. For each failed reference:
1. **WebSearch** for `"Title" author year` (try `site:arxiv.org` too)
2. **WebFetch** the paper page → extract title, authors, year, venue, DOI/ArXiv ID
3. If DOI/ArXiv found → retry `aii_semscholar_bib__fetch` with it
4. Last resort: write BibTeX by hand using **only verified info from the actual paper page**

---

### CLI (for manual use / debugging)

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-semscholar-bib" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_semscholar_bib__fetch.py --refs '[
  {"doi": "10.48550/arXiv.1706.03762", "author": "Vaswani", "year": 2017},
  {"arxiv": "2201.11903", "author": "Wei", "year": 2022},
  {"title": "Tree of Thoughts", "author": "Yao", "year": 2023}
]'
```

`--json, -j` — output raw JSON instead of .bib text

**If the script fails** with a connection error (ability server not running): create a local `.venv`, install server deps from `server_requirements.txt` into it, then import the `@aii_ability` function from the script and call it directly — bypassing the server:
```bash
uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r "$SKILL_DIR/scripts/server_requirements.txt"
```
````

### [5] SKILL-INPUT — aii-web-tools · 2026-06-22 07:56:15 UTC

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

### [6] SYSTEM-USER prompt · 2026-06-22 08:06:05 UTC

````
YOUR PREVIOUS SESSION WAS INTERRUPTED: A single operation exceeded the 720s message timeout. Each individual operation must complete within 720s. Do NOT mock, skip, or compromise your execution — still do the real work. Try to make operations run faster if possible. If a command genuinely takes longer than 720s, split it into sequential parts that each complete within the time limit.

Last messages before interruption:
  - [agent_tool_call] file_editor: Tool: file_editor
{
  "command": "view",
  "path": "/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/paper.tex",
  "file_text": null,
  "old_str": null,
  "new_str": null,
  "insert_line": null,
  "view_range": null
}
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "Here's the result of running `cat -n` on /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/paper.tex:\n     1\t\\documentclass[11pt,letterpaper]{article}\n     2\t\\us
  - [agent_response]: I notice a few typos in the paper:
1. "Interpretation" is misspelled as "Interpretaton" (line 241)
2. "Recommendations" is misspelled as "Recommendations" (line 245) - actually this looks correct
3. "Practitioners" should be "Practitioners" (line 249) - actually this is correct
4. "Corrected" is som
  - [agent_tool_call] file_editor: Tool: file_editor
{
  "command": "str_replace",
  "path": "/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/paper.tex",
  "file_text": null,
  "old_str": "\\subsection{Interpretaton of Results}",
  "new_str": "\\subsection{Interpretation of Results}",
  
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "No replacement was performed, old_str `\\subsection{Interpretaton of Results}` did not appear verbatim in /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/paper.tex."
  - [agent_response]: Let me search for the exact text in the file.
  - [agent_tool_call] terminal: Tool: terminal
{
  "command": "cd /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace && grep -n \"Interpret\" paper.tex",
  "is_input": false,
  "timeout": null,
  "reset": false
}
  - [agent_tool_result] terminal: Tool: terminal
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "252:\\subsection{Interpretation of Results}"}], "is_error": false, "command": "cd /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace && grep -n \"Interpret\" paper.tex", "
  - [agent_response]: I see - the text is "Interpretation" which is actually a valid (though less common) spelling. Let me check the paper more carefully for actual issues. Let me also search for other potential typos.
  - [agent_tool_call] terminal: Tool: terminal
{
  "command": "cd /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace && grep -n \"Corrected\\|Corrected\" paper.tex | head -20",
  "is_input": false,
  "timeout": null,
  "reset": false
}
  - [agent_tool_result] terminal: Tool: terminal
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "22:MinHash is a widely used algorithm for estimating Jaccard similarity in near-duplicate detection. However, quantifying uncertainty in MinHash estimates remains challenging, especially for short text documents whe
  - [agent_response]: I see "Corrected" is consistently spelled "Corrected" (with one 't') throughout the paper. This is actually a typo - it should be "Corrected" (with two 't's). Let me fix this throughout the paper. Also, let me check for "Comprehensive" which is spelled "Comprehensive" (missing 'h').

Let me fix thes
  - [agent_tool_call] file_editor: Tool: file_editor
{
  "command": "str_replace",
  "path": "/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/paper.tex",
  "file_text": null,
  "old_str": "Corrected Bootstrap",
  "new_str": "Corrected Bootstrap",
  "insert_line": null,
  "view_range": nu
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "Invalid `new_str` parameter: Corrected Bootstrap. No replacement was performed. `new_str` and `old_str` must be different."}], "is_error": true, "command": "str_replace", "path": null, "prev_exist": true, "old_co
  - [agent_response]: I see - the typo is actually "Corrected" (with two 't's) vs "Corrected" (which is the correct spelling). Let me search more carefully for the actual typo. Let me view the exact text around line 111.
  - [agent_tool_call] file_editor: Tool: file_editor
{
  "command": "view",
  "path": "/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/paper.tex",
  "file_text": null,
  "old_str": null,
  "new_str": null,
  "insert_line": null,
  "view_range": [
    108,
    120
  ]
}
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "Here's the result of running `cat -n` on /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/paper.tex:\n   108\t\n   109\tWe use a Beta prior on Jaccard similarity, upd
  - [agent_response]: I see - "Corrected" is spelled "Corrected" (with two 't's) which is correct. The issue is that "Datasets" is spelled "Datasets" (missing first 'a') on line 117. Let me fix this and also check for other typos.
  - [agent_tool_call] file_editor: Tool: file_editor
{
  "command": "str_replace",
  "path": "/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/paper.tex",
  "file_text": null,
  "old_str": "\\subsection{Datasets}",
  "new_str": "\\subsection{Datasets}",
  "insert_line": null,
  "view_rang
  - [agent_tool_result] file_editor: Tool: file_editor
Result:
{"content": [{"cache_prompt": false, "type": "text", "text": "Invalid `new_str` parameter: \\subsection{Datasets}. No replacement was performed. `new_str` and `old_str` must be different."}], "is_error": true, "command": "str_replace", "path": null, "prev_exist": true, "old

CONTINUE FOLLOWING THESE INSTRUCTIONS:

<workspace>
Your workspace: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/`:
GOOD: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/file.py`, `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace
</workspace>

<task>
Create a publication-ready top-conference LaTeX paper with BibTeX from <paper_text> and <available_figures>, compile to PDF.
</task>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<paper_text>
title: >-
  Why Extreme Value Theory Fails for MinHash Confidence Intervals on Short Text (and What to Do Instead)
abstract: >-
  MinHash is a widely used algorithm for estimating Jaccard similarity in near-duplicate detection. However, quantifying uncertainty
  in MinHash estimates remains challenging, especially for short text documents where the small number of shingles leads to
  high-variance estimates. We investigate whether Extreme Value Theory (EVT) can provide statistically principled confidence
  intervals for MinHash-based Jaccard estimates. Through theoretical analysis and comprehensive evaluation on 3,000 document
  pairs from real-world datasets (tweet_eval, ag_news), we show that the Gumbel and Weibull distributions—the theoretical
  limits for minima of i.i.d. random variables—provide poor fits to MinHash signature minima from short text documents (10-100
  shingles), with Kolmogorov-Smirnov test p-values < 10^{-20}. We identify three causes: finite-sample bias, dependence between
  overlapping shingles, and hash function discretization. To address this, we implement and evaluate five uncertainty quantification
  methods: EVT-Gumbel, EVT-Weibull, Corrected Bootstrap, Analytical Binomial (Clopper-Pearson), and Bayesian (Beta prior).
  Our evaluation on 3,000 document pairs shows that Analytical Binomial and Bayesian methods achieve 96.5% and 94.8% coverage
  (within 2% of the nominal 95% level), compared to 75.5% for Corrected Bootstrap with 1,000 resamples. Computation times
  are 84μs for Clopper-Pearson, 41μs for Wilson score, 395μs for Bayesian, and 2,000+μs for Bootstrap per confidence interval.
  We provide practitioner guidelines and open-source implementations for uncertainty quantification in MinHash-based near-duplicate
  detection.
paper_text: |-
  # Introduction

  Near-duplicate detection is a fundamental task in information retrieval, content moderation, and data cleaning. The MinHash algorithm [1] estimates Jaccard similarity between sets by comparing minimum hash values of their elements, providing a computationally efficient approach for large-scale deduplication. However, MinHash is fundamentally a statistical estimator, and its uncertainty quantification remains an open challenge, particularly for short text documents where the number of shingles (contiguous subsequences) is small.

  For short text documents such as tweets, SMS messages, and news headlines, the small number of shingles (typically 10-100) leads to high-variance Jaccard similarity estimates. Practitioners currently rely on heuristics or computationally expensive bootstrap methods to assess confidence in MinHash estimates. Bootstrap methods require resampling the data many times (typically 1,000-10,000 resamples) to construct confidence intervals, resulting in computational costs that scale as O(B*k) where B is the number of bootstrap samples and k is the number of hash functions.

  This paper explores whether Extreme Value Theory (EVT) can provide a theoretically grounded and computationally efficient alternative to bootstrap for uncertainty quantification in MinHash-based Jaccard similarity estimation. The core hypothesis is that MinHash signature values—being the minimum of uniform random variables—follow a Gumbel distribution (Extreme Value Type I) under the Fisher-Tippett-Gnedenko theorem [2]. If this hypothesis holds, it would enable closed-form confidence intervals and hypothesis tests for Jaccard similarity estimates with computational cost of only O(k) operations.

  ## Problem Statement

  Given two sets A and B, the Jaccard similarity is defined as:

  \[J(A, B) = \frac{|A \cap B|}{|A \cup B|}\]

  MinHash estimates Jaccard similarity by computing k independent hash functions and keeping the minimum hash value for each function. The probability that MinHash values match equals the Jaccard similarity [1]. However, for finite k and small set sizes, the MinHash estimate has variance that depends on the underlying distribution of hash values.

  The key research question is: *Does the distribution of MinHash signature minima follow a Gumbel distribution for short text documents, and if so, can we derive statistically principled confidence intervals for Jaccard similarity estimates?*

  ## Contributions

  This paper makes the following contributions:

  1. **Theoretical Analysis**: We derive the theoretical connection between MinHash and Extreme Value Theory, showing that the minimum of n i.i.d. uniform(0,1) variables converges to an exponential distribution as n→∞, which connects to the Gumbel distribution. We also identify the key assumptions (i.i.d., asymptotic regime) and their violations in practice.

  2. **Empirical Verification**: We implement MinHash with 128 hash functions using MD5 and empirically test whether MinHash signature minima follow Gumbel or Weibull distributions for short text documents (10-100 character shingles). We use real-world datasets from HuggingFace Hub (tweet_eval, ag_news) with 39,951 documents \footnote{Code: \url{https://github.com/AMGrobelnik/ai-invention-1ba2b3-why-extreme-value-theory-fails-for-minha/tree/main/round-1/dataset-1}}.

  3. **Distribution Fit Analysis**: We perform Kolmogorov-Smirnov (KS) tests and Anderson-Darling (AD) tests to evaluate Gumbel and Weibull distribution fits across three datasets and multiple shingle count ranges (10-30, 30-50, 50-100).

  4. **Negative Results on EVT**: We report that neither Gumbel nor Weibull distributions provide adequate fit for MinHash minima from short text documents, with all KS test p-values < 10^{-20}. This suggests that the asymptotic EVT approximation does not hold in the small-sample regime due to finite-sample bias, dependence between overlapping shingles, and hash function discretization.

  5. **Alternative UQ Methods**: We implement and evaluate two practical alternatives to EVT-MinHash: (a) analytical binomial confidence intervals (Clopper-Pearson exact and Wilson score) based on matching hash counts, and (b) Bayesian approach with Beta prior informed by document length \footnote{Code: \url{https://github.com/AMGrobelnik/ai-invention-1ba2b3-why-extreme-value-theory-fails-for-minha/tree/main/round-2/experiment-2}}.

  6. **Comprehensive Evaluation**: We compare five uncertainty quantification methods (EVT-Gumbel, EVT-Weibull, Corrected Bootstrap, Analytical Binomial, Bayesian) on 3,000 document pairs from real-world datasets. Results show that Analytical Binomial and Bayesian methods achieve 96.5% and 94.8% coverage (within 2% of 95% target), while Corrected Bootstrap only achieves 75.5% coverage \footnote{Code: \url{https://github.com/AMGrobelnik/ai-invention-1ba2b3-why-extreme-value-theory-fails-for-minha/tree/main/round-2/evaluation-1}}.

  7. **Computational Cost Analysis**: We benchmark all methods and show that analytical approaches are 10-50x faster than bootstrap, with Clopper-Pearson taking 84μs, Wilson score 41μs, Bayesian 395μs, and Bootstrap 2,000+μs per confidence interval.

  # Related Work

  ## MinHash and LSH

  MinHash was introduced by Broder [1] for estimating resemblance and containment between documents. The algorithm computes the minimum hash value for each set and uses the fraction of matching minima to estimate Jaccard similarity. Subsequent work has extended MinHash to weighted sets [3], improved memory efficiency using Bloom filters (LSHBloom [4]), and developed sampling-based approaches for Jaccard estimation [5].

  ## Uncertainty Quantification for MinHash

  Existing work on uncertainty quantification for MinHash includes:

  1. **FracMinHash** [6]: Derives confidence intervals for FracMinHash in bioinformatics applications using asymptotic normality. However, the normal approximation may fail for small sketches or extreme similarities.

  2. **Sampling-Based Jaccard Estimation** [5]: Introduces a binomial model for predicting overlap between random samples of sets, providing error bounds and sample size requirements. This approach requires storing random samples and is not directly applicable to MinHash sketches.

  3. **Bootstrap Methods** [7]: Bootstrap confidence intervals are a common approach for uncertainty quantification but require many resamples (B = 1,000-10,000) for accurate coverage, resulting in high computational cost.

  ## Extreme Value Theory in Sketching

  SetSketch [8] uses the Gumbel distribution for cardinality estimation (counting distinct elements) but not for Jaccard similarity confidence intervals. To the best of our knowledge, this is the first work to investigate whether MinHash signature values follow an extreme value distribution for the purpose of constructing confidence intervals for Jaccard similarity, and to provide a comprehensive comparison of alternative uncertainty quantification methods.

  # Methods

  ## MinHash Implementation

  We implement MinHash with k = 128 independent hash functions using MD5. Each hash function is generated with a different seed. For a document with text t, we extract character shingles of length k_shingle = 3. The Jaccard similarity estimate is the fraction of matching signature values.

  ## Extreme Value Theory Background

  The Fisher-Tippett-Gnedenko theorem states that the minimum of n i.i.d. random variables converges to one of three extreme value distributions. For uniform(0,1) variables, the limiting distribution is Gumbel.

  ### Assumptions and Limitations of EVT for MinHash

  The application of EVT to MinHash relies on: (1) i.i.d. hash values (violated by overlapping shingles), (2) asymptotic regime (n→∞, not satisfied for 10-100 shingles), (3) uniform hash functions (approximated by MD5), (4) continuous distribution (violated by discrete MD5 outputs).

  ## Alternative Uncertainty Quantification Methods

  ### Analytical Binomial Confidence Intervals

  The number of matching hashes out of k follows a binomial distribution. We implement Clopper-Pearson (exact) and Wilson score intervals.

  ### Bayesian Approach

  We use a Beta prior on Jaccard similarity, updated with observed match count. We use uniform prior (α=1, β=1) and length-informed prior.

  ### Corrected Bootstrap

  The bootstrap method resamples shingles (not signatures) to preserve dependence. For each resample, we recompute signatures and Jaccard estimate.

  # Experiments

  ## Datasets

  We evaluate on real-world datasets from HuggingFace Hub: Tweet Eval Sentiment (10K tweets, avg 68.5 shingles), Tweet Eval Emoji (10K tweets, avg 53.2 shingles), AG News (19,951 headlines, avg 156.8 shingles). Total: 39,951 documents .

  [FIGURE:fig1]

  ## Experimental Setup

  We generate 3,000 document pairs, compute true and estimated Jaccard, fit EVT distributions, and evaluate five UQ methods.

  # Results

  ## Distribution Fit Results

  Table 1 shows KS test results. All p-values < 10^{-20}, indicating poor fit.

  **Table 1**: Distribution fit results. All KS p-values < 10^{-20}.

  [FIGURE:fig2]

  ## Comprehensive Evaluation of UQ Methods

  Table 2 shows coverage probability for 95% CIs.

  **Table 2**: Coverage probability. Analytical Binomial: 96.5%, Bayesian: 94.8%, Corrected Bootstrap: 75.5%.

  ## Confidence Interval Width

  **Table 3**: CI width. Analytical Binomial: mean=0.0996, Bayesian: mean=0.0935, Corrected Bootstrap: mean=0.0543.

  ## Bias and RMSE

  **Table 4**: Bias and RMSE. All methods have similar bias (~0.0026) except Bayesian (0.0091).

  ## Computational Cost

  **Table 5**: Computational cost. Clopper-Pearson: 84μs, Wilson: 41μs, Bayesian: 395μs, Bootstrap: 2,000+μs.

  [FIGURE:fig3]

  # Discussion

  ## Interpretation of Results

  EVT distributions provide poor fit due to: (1) small sample size, (2) dependence between shingles, (3) hash function bias, (4) discrete distribution. Despite this, EVT methods achieve acceptable coverage, suggesting robustness but lacking theoretical justification.

  ## Recommendations for Practitioners

  1. Use Analytical Binomial or Bayesian for well-calibrated CIs.
  2. Avoid EVT for short text (<100 shingles).
  3. Increase bootstrap resamples if using bootstrap.
  4. Method selection: Speed → Wilson Score (41μs), Rigor → Clopper-Pearson, Priors → Bayesian.

  # Conclusion

  We show that EVT distributions provide poor fits to MinHash minima from short text. Analytical Binomial and Bayesian methods achieve 96.5% and 94.8% coverage. Analytical methods are 10-50x faster than bootstrap. We provide practitioner guidelines and open-source implementations.

  # Acknowledgments

  This work was supported by the AI Inventor automated research system.

  # References

  [1] Broder, A. Z. (1997). On the resemblance and containment of documents.
  [2] Fisher & Tippett (1928). Limiting forms of the frequency distribution.
  [3] Wu et al. (2018). A review for weighted MinHash algorithms.
  [4] Khan et al. (2024). LSHBloom. arXiv:2411.04257.
  [5] Joshi (2025). Sampling-Based Estimation of Jaccard. arXiv:2507.10019.
  [6] Irber & Brown (2023). Debiasing FracMinHash. PLoS Comput Biol.
  [7] Efron & Tibshirani (1994). An introduction to the bootstrap.
  [8] Ertl (2021). SetSketch. VLDB.
summary: >-
  This paper investigates EVT for uncertainty quantification in MinHash-based Jaccard estimation. We show that EVT distributions
  provide poor fits to MinHash minima from short text (KS p-values < 10^{-20}). We implement and evaluate five UQ methods
  on 3,000 document pairs. Results show that Analytical Binomial and Bayesian methods achieve 96.5% and 94.8% coverage, while
  Corrected Bootstrap only achieves 75.5%. Analytical methods are 10-50x faster than bootstrap.
</paper_text>

<available_figures>
--- Item 1 ---
id: fig1
title: Dataset Shingle Count Distributions
caption: Distribution of shingle counts across three datasets.
image_gen_detailed_description: Three panel histogram. White background, sans-serif font.
aspect_ratio: '21:9'
summary: Shingle count distributions
figure_path: figures/fig1_v0.jpg

--- Item 2 ---
id: fig2
title: 'QQ Plots: EVT Distribution Fit'
caption: QQ plots showing poor fit of EVT distributions to MinHash minima.
image_gen_detailed_description: Two panel QQ plot. White background, sans-serif font.
aspect_ratio: '21:9'
summary: QQ plots showing poor EVT fit
figure_path: figures/fig2_v0.jpg

--- Item 3 ---
id: fig3
title: Computational Cost Comparison
caption: Computational cost per CI for five UQ methods.
image_gen_detailed_description: Grouped bar chart, log scale. White background, sans-serif font.
aspect_ratio: '21:9'
summary: Computational cost comparison
figure_path: figures/fig3_v0.jpg
</available_figures>

<figure_requirements>
CRITICAL: Include ALL figures from <available_figures>. No exceptions.

- Every figure MUST use \includegraphics{figures/filename.jpg}
- Do NOT skip, convert to tables, or describe without inserting
- Each needs: \begin{figure*|figure}[placement], \includegraphics, \caption, \label, \end{...} — pick env + placement by the figure's `aspect_ratio` field (see PLACEMENT below). Constrain every \includegraphics with `width=\linewidth,height=0.4\textheight,keepaspectratio` (single-column) or `width=\textwidth,height=0.45\textheight,keepaspectratio` (figure*). Use exactly these option keys — `max height=` is NOT valid LaTeX
- Use the `caption` field from each figure for \caption{...} — do NOT invent new captions
- Place figures where their [FIGURE:fig_id] markers appear in paper_text
- VERIFICATION: paper.tex MUST have exact same number of \includegraphics as <available_figures>
- Do NOT generate new figure images (no matplotlib, no PIL, no image generation). Use ONLY the pre-generated figures from <available_figures>. They were already created by a previous pipeline step.

PLACEMENT BY ASPECT RATIO (use the `aspect_ratio` field on each figure):
- `21:9` (architecture diagrams / hero figures): \begin{figure*}[!t] (full two-column width, top of page). The hero architecture diagram should appear EARLY in the paper — typically at the top of page 2. Marker placement in paper_text already determines this; preserve it.
- `16:9` (comparisons, multi-panel results): \begin{figure*}[!t] for full-width or \begin{figure}[!htbp] for single-column.
- `4:3` / `1:1` / `3:2` / `3:4` / `9:16`: \begin{figure}[!htbp] (single-column).
</figure_requirements>

<artifact_links>
The paper_text contains \footnote{Code: \url{...}} references linking to artifact source code
on GitHub. Include \usepackage{hyperref} and \usepackage{url}.
Preserve these exactly as-is — do not remove, rewrite, or convert them to plain text.
The URLs will not resolve yet (the repo is deployed after compilation) — do NOT try to verify or fix them.
</artifact_links>

<headings>
NEVER use inline math (``$...$``) inside ``\section{...}`` / ``\subsection{...}`` / ``\subsubsection{...}`` arguments — hyperref's bookmark builder errors out (``Token not allowed in a PDF string``) and the PDF outline breaks. If a section heading needs a math-looking term, use the text equivalent (``d star`` not ``$d^*$``, ``alpha-equivalent`` not ``$\alpha$-equivalent``) or wrap it in ``\texorpdfstring{$math$}{plain}``. Inline math inside body paragraphs is fine.
</headings>

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Read and STRICTLY follow these skills: aii-paper-to-latex, aii-semscholar-bib.
TODO 2. Review <paper_text> and <available_figures>. Copy all figure images into ./figures/ in your workspace. Count figures — MUST include every one. Plan placements per section. Build `./references.bib` via aii_semscholar_bib__fetch — collect DOIs/ArXiv IDs from <paper_text> and batch-fetch all BibTeX in one call. Do NOT fabricate entries.
TODO 3. Create `./paper.tex` per aii-paper-to-latex skill's setup, write ALL sections, insert ALL figures from <available_figures>, include `./references.bib` via \bibliography. Compile to PDF per skill's process. Fix errors.
TODO 4. CRITICAL VERIFICATION: Run `grep -c 'includegraphics' paper.tex`, confirm count equals figures in <available_figures>. If not, add missing figures. Verify `./paper.pdf` was created.
TODO 5. VISUAL REVIEW: Write Python script to convert EVERY page of paper.pdf to PNG at 150 DPI (use pdf2image or pymupdf). Then read ALL page screenshots — each page image costs ~1,600 tokens so a 15-page paper is only ~24K tokens. You MUST read every page. The ONLY exception is if all page images would not fit in your remaining context — in that case, read as many as fit and state which pages you are skipping and why. Check every page for layout issues, overlapping figures, cut-off text, bad spacing, formatting problems. Fix issues and recompile.
TODO 6. FINAL READ: Check page count (`pdfinfo paper.pdf` or pymupdf). Read entire paper.pdf — check for missing sections, unclear explanations, inconsistencies, typos. Fix and recompile. The ONLY exception is if all pages would not fit in your remaining context — in that case, read as many pages as fit and state which pages you are skipping and why.
</todos>

---

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "FullPaperExpectedFiles": {
      "description": "All expected output files from full paper generation.",
      "properties": {
        "paper_tex_path": {
          "description": "Path to LaTeX source file. Example: 'paper.tex'",
          "title": "Paper Tex Path",
          "type": "string"
        },
        "paper_pdf_path": {
          "description": "Path to compiled PDF. Example: 'paper.pdf'",
          "title": "Paper Pdf Path",
          "type": "string"
        },
        "references_bib_path": {
          "description": "Path to BibTeX bibliography file. Example: 'references.bib'",
          "title": "References Bib Path",
          "type": "string"
        },
        "figure_paths": {
          "description": "Paths to all figure image files. Example: ['figures/fig1_v0.jpg', 'figures/fig2_v0.jpg']",
          "items": {
            "type": "string"
          },
          "title": "Figure Paths",
          "type": "array"
        }
      },
      "required": [
        "paper_tex_path",
        "paper_pdf_path",
        "references_bib_path",
        "figure_paths"
      ],
      "title": "FullPaperExpectedFiles",
      "type": "object"
    }
  },
  "description": "Full paper \u2014 structured output from paper generation.",
  "properties": {
    "title": {
      "description": "Short descriptive title for this paper generation task (roughly 30-90 characters)",
      "maxLength": 90,
      "minLength": 30,
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "description": "Brief summary of the generated paper: sections written, figures included, compilation status",
      "maxLength": 5000,
      "minLength": 500,
      "title": "Summary",
      "type": "string"
    },
    "out_expected_files": {
      "$ref": "#/$defs/FullPaperExpectedFiles",
      "description": "All output files you created. Must include paper.tex, paper.pdf, references.bib, and paths to all figure files."
    }
  },
  "required": [
    "title",
    "summary",
    "out_expected_files"
  ],
  "title": "FullPaper",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/4_gen_paper_repo/_4_assemble_paper/paper/workspace/.sdk_openhands_agent_struct_out.json`.

Build and evaluate a simple MinHash near-duplicate detector for short text documents.
````

### [7] SYSTEM-USER prompt · 2026-06-22 08:10:22 UTC

```
<validation-feedback>
Attempt 1 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `title`: 'Why Extreme Value Theory Fails for MinHash Confidence Intervals on Short Text (and What to Do Instead)' is too long (at most 90 characters, got 102)
Every required field must be present and every field type must match the schema.

Please use the Write tool to overwrite `.sdk_openhands_agent_struct_out.json` with corrected JSON. Do not invent new fields; match the schema you were given.
</validation-feedback>
```
