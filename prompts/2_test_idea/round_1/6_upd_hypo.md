# upd_hypo — test_idea

> Phase: `invention_loop` · round 1 · `upd_hypo`
> Run: `run_EqcgJR2naF4b` — Why Extreme Value Theory Fails for MinHash Confidence Intervals on Short Text (and What to Do Instead)
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `upd_hypo` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 05:42:04 UTC

````
<current_hypothesis>
The hypothesis as it stands. Revise it based on the evidence below.

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
</current_hypothesis>

<all_artifacts>
Complete set of research artifacts across all iterations.

--- Item 1 ---
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

--- Item 2 ---
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

--- Item 3 ---
id: art_SiAaMxtNmflp
type: experiment
title: >-
  EVT-MinHash Distribution Verification and Bootstrap Baseline for Short Text Documents
summary: >-
  This experiment empirically verified whether MinHash signature minima follow Gumbel or Weibull distributions for short text
  documents (10-100 character shingles). The implementation includes: (1) MinHash with 128 hash functions using MD5, (2) Gumbel
  and Weibull distribution fitting with KS-test goodness-of-fit, (3) Bootstrap confidence intervals for Jaccard similarity
  with 1000 resamples, (4) Computational cost comparison. Key findings: Neither Gumbel nor Weibull distributions fit well
  (KS p-values < 0.05), suggesting EVT assumptions may not hold for MinHash with short text. The experiment processed 2000
  synthetic documents across 3 datasets (tweets, SMS, headlines) with 1000 document pairs. Output includes distribution fit
  results (6 entries), bootstrap CI results (600 pairs), computational cost metrics, and 6 visualization files (QQ plots,
  histograms).
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json
</all_artifacts>

<new_artifacts_this_iteration>
These 3 artifacts were created THIS iteration.

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

id: art_SiAaMxtNmflp
type: experiment
title: >-
  EVT-MinHash Distribution Verification and Bootstrap Baseline for Short Text Documents
summary: >-
  This experiment empirically verified whether MinHash signature minima follow Gumbel or Weibull distributions for short text
  documents (10-100 character shingles). The implementation includes: (1) MinHash with 128 hash functions using MD5, (2) Gumbel
  and Weibull distribution fitting with KS-test goodness-of-fit, (3) Bootstrap confidence intervals for Jaccard similarity
  with 1000 resamples, (4) Computational cost comparison. Key findings: Neither Gumbel nor Weibull distributions fit well
  (KS p-values < 0.05), suggesting EVT assumptions may not hold for MinHash with short text. The experiment processed 2000
  synthetic documents across 3 datasets (tweets, SMS, headlines) with 1000 document pairs. Output includes distribution fit
  results (6 entries), bootstrap CI results (600 pairs), computational cost metrics, and 6 visualization files (QQ plots,
  histograms).
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json
</new_artifacts_this_iteration>

<current_paper>
The paper draft from this iteration — represents the current state of the research story.

# Introduction

Near-duplicate detection is a fundamental task in information retrieval, content moderation, and data cleaning. The MinHash algorithm [1] estimates Jaccard similarity between sets by comparing minimum hash values of their elements, providing a computationally efficient approach for large-scale deduplication. However, MinHash is fundamentally a statistical estimator, and its uncertainty quantification remains an open challenge, particularly for short text documents where the number of shingles (contiguous subsequences) is small.

For short text documents such as tweets, SMS messages, and news headlines, the small number of shingles (typically 10-100) leads to high-variance Jaccard similarity estimates. Practitioners currently rely on heuristics or computationally expensive bootstrap methods to assess confidence in MinHash estimates. Bootstrap methods require resampling the data many times (typically 1000-10000 resamples) to construct confidence intervals, resulting in computational costs that scale as O(B×k) where B is the number of bootstrap samples and k is the number of hash functions.

This paper explores whether Extreme Value Theory (EVT) can provide a theoretically grounded and computationally efficient alternative to bootstrap for uncertainty quantification in MinHash-based Jaccard similarity estimation. The core hypothesis is that MinHash signature values—being the minimum of uniform random variables—follow a Gumbel distribution (Extreme Value Type I) under the Fisher-Tippett-Gnedenko theorem [2]. If this hypothesis holds, it would enable closed-form confidence intervals and hypothesis tests for Jaccard similarity estimates with computational cost of only O(k) operations.

## Problem Statement

Given two sets A and B, the Jaccard similarity is defined as:

\[J(A, B) = \frac{|A \cap B|}{|A \cup B|}\]

MinHash estimates Jaccard similarity by computing k independent hash functions and keeping the minimum hash value for each function. The probability that MinHash values match equals the Jaccard similarity [1]. However, for finite k and small set sizes, the MinHash estimate has variance that depends on the underlying distribution of hash values.

The key research question is: *Does the distribution of MinHash signature minima follow a Gumbel distribution for short text documents, and if so, can we derive statistically principled confidence intervals for Jaccard similarity estimates?*

## Contributions

This paper makes the following contributions:

1. **Theoretical Analysis**: We derive the theoretical connection between MinHash and Extreme Value Theory, showing that the minimum of n i.i.d. uniform(0,1) variables converges to an exponential distribution as n→∞, which connects to the Gumbel distribution [ARTIFACT:art_T61Vq8hqL41x].

2. **Empirical Verification**: We implement MinHash with 128 hash functions using MD5 and empirically test whether MinHash signature minima follow Gumbel or Weibull distributions for short text documents (10-100 character shingles) [ARTIFACT:art_SiAaMxtNmflp].

3. **Distribution Fit Analysis**: We perform Kolmogorov-Smirnov (KS) tests and Anderson-Darling (AD) tests to evaluate Gumbel and Weibull distribution fits across three datasets (tweets, SMS, headlines) and two shingle count ranges (10-30, 30-50).

4. **Bootstrap Baseline**: We implement bootstrap confidence intervals with 1000 resamples and compare computational costs against the proposed EVT-MinHash approach.

5. **Negative Results**: We report that neither Gumbel nor Weibull distributions provide adequate fit for MinHash minima from short text documents, with all KS test p-values < 0.05. This suggests that the asymptotic EVT approximation does not hold in the small-sample regime.

# Related Work

## MinHash and LSH

MinHash was introduced by Broder [1] for estimating resemblance and containment between documents. The algorithm computes the minimum hash value for each set and uses the fraction of matching minima to estimate Jaccard similarity. Subsequent work has extended MinHash to weighted sets [3], improved memory efficiency using Bloom filters (LSHBloom [4]), and developed sampling-based approaches for Jaccard estimation [5].

## Uncertainty Quantification for MinHash

Existing work on uncertainty quantification for MinHash includes:

1. **FracMinHash** [6]: Derives confidence intervals for FracMinHash in bioinformatics applications using asymptotic normality. However, the normal approximation may fail for small sketches or extreme similarities.

2. **Sampling-Based Jaccard Estimation** [5]: Introduces a binomial model for predicting overlap between random samples of sets, providing error bounds and sample size requirements. This approach requires storing random samples and is not directly applicable to MinHash sketches.

3. **Bootstrap Methods** [7]: Bootstrap confidence intervals are a common approach for uncertainty quantification but require many resamples (B = 1000-10000) for accurate coverage, resulting in high computational cost.

## Extreme Value Theory in Sketching

SetSketch [8] uses the Gumbel distribution for cardinality estimation (counting distinct elements) but not for Jaccard similarity confidence intervals. To the best of our knowledge, this is the first work to investigate whether MinHash signature values follow an extreme value distribution for the purpose of constructing confidence intervals for Jaccard similarity.

# Methods

## MinHash Implementation

We implement MinHash with k = 128 independent hash functions using MD5. Each hash function is generated with a different seed:

\[h_i(x) = \text{MD5}(\text{seed}_i \| x)\]

where seed_i = seed + i × 1000 for i = 0, ..., k-1.

For a document with text t, we extract character shingles of length k_shingle = 3. The MinHash signature is computed as:

\[m(t) = [\min_{s \in \text{shingles}(t)} h_0(s), ..., \min_{s \in \text{shingles}(t)} h_{k-1}(s)]\]

The Jaccard similarity estimate is:

\[\hat{J}(t_1, t_2) = \frac{1}{k} \sum_{i=0}^{k-1} \mathbb{I}(m_i(t_1) = m_i(t_2))\]

## Extreme Value Theory Background

The Fisher-Tippett-Gnedenko theorem states that the minimum of n i.i.d. random variables, after proper renormalization, converges in distribution to one of three extreme value distributions: Gumbel, Fréchet, or Weibull [2]. For the minimum case with a uniform(0,1) distribution (corresponding to uniform hash functions in MinHash), the limiting distribution is Gumbel.

Specifically, if X₁, ..., Xₙ are i.i.d. uniform(0,1), then n·min(X₁, ..., Xₙ) converges to an exponential distribution with rate 1 as n→∞. Through the transformation Y = -ln(X), this connects to the Gumbel distribution: if Y ~ Exponential(1), then -ln(Y) ~ Gumbel(0,1).

The Gumbel distribution (for minima) has CDF:

\[F(x; \mu, \beta) = 1 - \exp(-\exp(-(x-\mu)/\beta))\]

with location parameter μ and scale parameter β.

## Distribution Fitting

We fit both Gumbel and Weibull distributions to MinHash signature minima using maximum likelihood estimation (MLE) via SciPy's `gumbel_l` and `weibull_min` functions. We evaluate the goodness-of-fit using:

1. **Kolmogorov-Smirnov (KS) test**: Compares the empirical CDF to the theoretical CDF. The KS statistic is:
   \[D_n = \sup_x |F_n(x) - F(x)|\]
   where F_n is the empirical CDF and F is the theoretical CDF. The p-value indicates whether the data could reasonably come from the theoretical distribution.

2. **Anderson-Darling (AD) test**: A more sensitive test that gives more weight to the tails of the distribution. We report the AD statistic and critical values at significance levels 25%, 10%, 5%, 2.5%, and 1%.

## Bootstrap Confidence Intervals

We implement percentile bootstrap confidence intervals for Jaccard similarity. Given two documents t₁ and t₂, we:

1. Compute the MinHash signature for each document
2. For b = 1, ..., B (B = 1000):
   a. Resample k hash functions with replacement
   b. Compute Jaccard estimate using resampled signatures
3. Compute the 2.5th and 97.5th percentiles of the bootstrap distribution for a 95% confidence interval

## Computational Cost Analysis

We measure the computational cost of:
1. MinHash signature computation (one document)
2. Bootstrap CI computation (100 resamples, extrapolated to 1000)

All experiments run on a machine with 6 CPUs and 31.0 GB RAM.

# Experiments

## Datasets

We evaluate on synthetic short text datasets generated to mimic real-world distributions:

1. **Tweets**: 2000 synthetic tweets with 10-50 words (avg 19.2 words)
2. **SMS**: 2000 synthetic SMS messages with 10-50 words (avg 11.8 words)
3. **Headlines**: 2000 synthetic news headlines with 10-50 words (avg 38.7 words)

Each dataset contains documents with varying shingle counts (number of character-3 shingles). We group documents into shingle count ranges: 10-30, 30-50, 50-70, 70-100.

[ARTIFACT:art_qBzTyA1I1xtt] describes the dataset collection and standardization process.

## Experimental Setup

For each dataset and shingle count range:
1. Filter documents to the specified shingle count range
2. Generate all document pairs (up to 1000 pairs)
3. Compute MinHash signatures for all documents (k = 128 hash functions)
4. Fit Gumbel and Weibull distributions to MinHash signature minima
5. Perform KS-test and AD-test for goodness-of-fit
6. Compute bootstrap confidence intervals for 100 document pairs
7. Measure computational cost

## Configuration

- Number of hash functions: k = 128
- Shingle type: character-3 shingles
- Number of bootstrap resamples: B = 1000
- Confidence level: 95%
- Number of document pairs: 1000
- Random seed: 42

# Results

## Distribution Fit Results

Table 1 shows the distribution fit results for all datasets and shingle count ranges. We report the selected distribution (Gumbel or Weibull), the KS statistic, KS p-value, and AD statistic.

**Key Finding**: Neither Gumbel nor Weibull distributions provide adequate fit for MinHash signature minima. All KS test p-values are < 0.05, indicating that we reject the null hypothesis that the data comes from the specified distribution.

| Dataset | Shingle Range | Selected | KS Statistic | KS p-value | AD Statistic |
|---------|---------------|----------|--------------|------------|--------------|
| tweets  | 10-30         | Gumbel   | 0.176        | 1.58e-28   | 53.59        |
| tweets  | 30-50         | Gumbel   | 0.304        | 4.01e-164  | 244.00       |
| sms     | 10-30         | Weibull  | 0.480        | 3.55e-275  | -            |
| sms     | 30-50         | Weibull  | 0.246        | 4.47e-107  | -            |
| headlines | 10-30      | Gumbel   | 0.235        | 2.82e-51   | 63.35        |
| headlines | 30-50      | Weibull  | 0.151        | 1.75e-40   | -            |

The KS p-values are extremely small (all < 1e-28), providing strong evidence against the Gumbel and Weibull distribution assumptions. The AD statistics are also large (50-244), further confirming poor fit.

[FIGURE:fig1]

Figure 1 shows QQ plots comparing the empirical distribution of MinHash signature minima to the fitted Gumbel and Weibull distributions. The systematic deviations from the diagonal line indicate poor fit for both distributions.

## Detailed Distribution Fit Analysis

For tweets (10-30 shingles):
- Gumbel fit: loc = 0.000511, scale = 0.000163, KS p-value = 1.58e-28
- Weibull fit: KS p-value = 1.51e-44 (even worse fit)

For tweets (30-50 shingles):
- Gumbel fit: loc = 0.000332, scale = 0.000206, KS p-value = 4.01e-164
- Weibull fit: KS p-value = 1.19e-182 (even worse fit)

The KS p-values become even smaller as the number of samples increases (from 1032 to 2000), indicating that the lack of fit is not due to small sample size.

## Bootstrap Confidence Intervals

Table 2 shows sample bootstrap confidence intervals for document pairs from the tweets dataset (10-30 shingles). The bootstrap CIs are wide, reflecting the high variance of Jaccard estimates for short text documents.

| Pair | Exact Jaccard | MinHash Estimate | Bootstrap CI (95%) |
|------|---------------|------------------|--------------------|
| 0    | 0.714         | 0.672            | [0.207, 0.520]    |
| 1    | 0.828         | 0.844            | [0.250, 0.583]    |
| 2    | 0.833         | 0.844            | [0.240, 0.583]    |

Note: The bootstrap CIs in Table 2 appear to be incorrectly computed (the intervals do not contain the MinHash estimate). This suggests an implementation issue with the bootstrap procedure that requires further investigation.

## Computational Cost

Table 3 shows the computational cost comparison between MinHash signature computation and bootstrap confidence interval computation.

| Operation | Time per Document (seconds) |
|-----------|----------------------------|
| MinHash signature | 0.0073 |
| Bootstrap CI (100 resamples) | 0.0037 |
| Bootstrap CI (1000 resamples, extrapolated) | 0.0365 |

The computational cost of bootstrap scales linearly with the number of resamples B. For B = 1000, bootstrap CI computation takes ~0.037 seconds per document pair, which is ~5× slower than MinHash signature computation alone.

If EVT-MinHash could be used, the computational cost would be O(k) for parameter estimation (via method of moments) plus O(1) for confidence interval computation (via delta method), compared to O(B×k) for bootstrap.

# Discussion

## Interpretation of Results

Our experimental results show that neither Gumbel nor Weibull distributions provide adequate fit for MinHash signature minima from short text documents. The KS test p-values are all < 0.05, and most are < 1e-20, providing overwhelming evidence against the EVT distributional assumptions.

There are several possible explanations for this negative result:

1. **Small Sample Size**: The Fisher-Tippett-Gnedenko theorem is an asymptotic result that holds as n→∞. For short text documents with 10-100 shingles, n may not be large enough for the Gumbel approximation to be accurate.

2. **Dependence**: The theorem assumes i.i.d. random variables. However, shingles in text documents are overlapping and not independent (e.g., "the cat" and "cat sat" share "cat"), violating the independence assumption.

3. **Hash Function Bias**: The theorem assumes truly uniform random variables. Real hash functions (MD5) approximate uniformity but may have biases, especially for small input spaces.

4. **Discrete Distribution**: MinHash signature values are discrete (integer MD5 outputs), while the Gumbel distribution is continuous. This discretization may affect the distributional fit.

## Comparison to Related Work

Our negative results contrast with SetSketch [8], which successfully uses the Gumbel distribution for cardinality estimation. The key difference is that cardinality estimation uses the maximum of many random variables (one per distinct element), while MinHash uses the minimum of a small number of shingles. The asymptotic approximation may be more accurate for cardinality estimation where n (number of distinct elements) is large.

## Limitations

This study has several limitations:

1. **Synthetic Data**: We use synthetic datasets because real-world datasets (tweet_eval, ag_news) failed to load due to HuggingFace issues. Synthetic data may not capture the full complexity of real text.

2. **Character Shingles**: We use character-3 shingles, which may not be optimal for all text types. Word shingles or k-shingles with different k values may produce different results.

3. **MD5 Hash Function**: We use MD5 for hashing, which may have different distributional properties than other hash functions (e.g., MurmurHash).

4. **Implementation Issues**: The bootstrap CI implementation appears to have bugs (CIs do not contain the point estimate). This requires further investigation.

## Future Work

Based on these negative results, we identify several directions for future work:

1. **Alternative EVT Approaches**: Investigate whether the generalized extreme value (GEV) distribution provides better fit than Gumbel or Weibull. The GEV distribution combines all three extreme value families and may be more flexible.

2. **Normalization Techniques**: Apply normalization techniques to improve the Gumbel approximation for finite sample sizes. The Fisher-Tippett-Gnedenko theorem requires renormalization (subtracting location, dividing by scale) for convergence.

3. **Bayesian Approaches**: Develop Bayesian confidence intervals for Jaccard similarity using posterior sampling or variational inference, which may be more accurate than bootstrap for small samples.

4. **Adaptive Methods**: Design adaptive methods that switch between EVT-based and bootstrap-based confidence intervals depending on the document length and shingle count.

# Conclusion

This paper investigated the hypothesis that MinHash signature values follow a Gumbel distribution under Extreme Value Theory, which would enable computationally efficient confidence intervals for Jaccard similarity estimates. Through theoretical analysis and empirical verification on synthetic short text datasets, we demonstrated that neither Gumbel nor Weibull distributions provide adequate fit for MinHash minima from short text documents (10-100 shingles).

Our key findings are:

1. **Negative Distribution Fit**: KS-test p-values are all < 0.05 (most < 1e-20), indicating that MinHash signature minima do not follow Gumbel or Weibull distributions for short text documents.

2. **Computational Cost**: Bootstrap confidence intervals require O(B×k) operations versus O(k) for proposed EVT-MinHash, but the distributional assumption fails to hold.

3. **Implementation and Evaluation**: We provide a complete open-source implementation of MinHash with distribution fitting and bootstrap baseline, enabling reproducibility and further research.

These negative results are valuable for the research community because they clarify the limitations of EVT-based approaches for MinHash uncertainty quantification in the small-sample regime. Future work should explore alternative approaches such as Bayesian methods, adaptive techniques, or normalization strategies to improve the accuracy of confidence intervals for Jaccard similarity estimates.

# References

[1] Broder, A. Z. (1997). On the resemblance and containment of documents. In Proceedings of the Compression and Complexity of Sequences.

[2] Fisher, R. A., & Tippett, L. H. C. (1928). Limiting forms of the frequency distribution of the largest or smallest member of a sample. Mathematical Proceedings of the Cambridge Philosophical Society, 24(2), 180-190.

[3] Wu, W., Li, B., Chen, L., Gao, J., & Zhang, C. (2018). A review for weighted MinHash algorithms. IEEE Transactions on Knowledge and Data Engineering, 34(6), 2553-2573.

[4] Khan, A., Underwood, R., Siebenschuh, C., Babuji, Y., Ajith, A., Hippe, K., Gökdemir, O., Brace, A., Chard, K., & Foster, I. T. (2024). LSHBloom: Memory-efficient, Extreme-scale Document Deduplication. arXiv preprint arXiv:2411.04257.

[5] Joshi, P. (2025). Sampling-Based Estimation of Jaccard Containment and Similarity. arXiv preprint arXiv:2507.10019.

[6] Irber, L., & Brown, C. T. (2023). Debiasing FracMinHash and deriving confidence intervals. PLoS Computational Biology, 19(1), e1010774.

[7] Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap. Chapman and Hall/CRC.

[8] Ertl, O. (2021). SetSketch: Filling the gap between MinHash and HyperLogLog. Proceedings of the VLDB Endowment, 14(11), 2244-2256.

# Acknowledgments

This work was supported by the AI Inventor automated research system. We thank the open-source community for providing the datasets, tools, and libraries used in this research.

</current_paper>

<reviewer_feedback>
Feedback from the paper reviewer this iteration.

- [MAJOR] (evidence) Discrepancy between paper claims and dataset artifact: The paper states 'We use synthetic datasets because real-world datasets (tweet_eval, ag_news) failed to load due to HuggingFace issues' (Limitations section). However, the dataset artifact art_qBzTyA1I1xtt was explicitly designed to collect these datasets from HuggingFace, and its summary claims success: 'Collected and standardized 3 short text datasets from HuggingFace Hub... Total documents: 39,951.' This discrepancy suggests either the data collection succeeded but the experiment code didn't use it (fell back to synthetic), or the dataset artifact claims are inaccurate. Either way, this needs resolution.
  Action: Investigate and resolve the data discrepancy: (1) Check if the HuggingFace datasets were actually collected in art_qBzTyA1I1xtt (examine data_out.json or full_data_out.json), (2) If real data exists, update the experiment code to use it instead of synthetic data, (3) If real data doesn't exist, update the dataset artifact summary to be accurate and provide a more detailed explanation of why HuggingFace loading failed (specific error messages, not just 'issues'). The paper should use real data if possible, as synthetic data weakens the evaluation.
- [MAJOR] (methodology) Buggy bootstrap CI implementation: Table 2 shows bootstrap confidence intervals that do NOT contain the MinHash estimate or the exact Jaccard similarity. For example, Pair 0 has Exact Jaccard=0.714, MinHash Estimate=0.672, but Bootstrap CI=[0.207, 0.520]. This indicates a serious bug in the bootstrap implementation. The paper acknowledges this in a note, but still includes the table. This undermines the computational cost comparison (Section 'Computational Cost') which relies on the bootstrap implementation being correct.
  Action: Fix the bootstrap CI implementation: (1) Debug the bootstrap code in method.py - the issue is likely in how bootstrap resamples are generated or how CIs are computed. Bootstrap for MinHash should involve resampling shingles (or hash functions) and recomputing Jaccard estimates, not resampling the signatures directly. (2) Recompute Table 2 with correct CIs. (3) Verify that CIs have proper coverage properties (e.g., 95% CI should contain true Jaccard ~95% of the time in simulation).
- [MAJOR] (novelty) Limited contribution due to negative results: The paper's main finding is that EVT does NOT work for MinHash confidence intervals in the small-sample regime. While negative results are valuable, this paper's contribution is primarily 'this approach doesn't work' without providing: (a) alternative solutions that DO work, (b) deep insights into WHY it fails beyond high-level reasons (small sample, dependence, hash bias), or (c) actionable guidelines for practitioners. At a top-tier venue, a paper with negative results typically needs to provide significant insights or open new research directions to be accepted.
  Action: Strengthen the contribution beyond negative results by adding one or more of: (1) A corrected EVT approach (e.g., using proper normalization, GEV distribution instead of just Gumbel/Weibull, or finite-sample corrections), (2) A practical alternative for MinHash uncertainty quantification (e.g., analytical variance bounds based on the binomial model, Bayesian approach with prior on Jaccard), (3) Deeper analysis of why EVT fails - derive the exact finite-sample distribution, quantify the dependence bias, analyze hash function properties, (4) Actionable guidelines: when does EVT work? (e.g., for longer documents, different hash functions, or after normalization).
- [MINOR] (rigor) Incomplete results in Table 1: The table shows KS statistics and p-values, but some rows have incomplete information. For example, the 'sms 10-30' and 'sms 30-50' rows show Weibull as 'Selected' but don't show the Weibull KS p-value in the main table (it's only in the detailed analysis section). The AD statistic column shows '-' for Weibull fits without explanation. A reader should be able to understand the full comparison from Table 1 without reading the text.
  Action: Complete Table 1: (1) Add Weibull KS p-values to all rows where Weibull was evaluated, (2) Either add AD statistics for Weibull fits or explain why they're not reported (e.g., 'AD test not standard for Weibull in SciPy'), (3) Consider adding columns for both Gumbel and Weibull fit statistics to make comparison easier, or create separate tables for each distribution.
- [MINOR] (clarity) Confusion about EVT application to MinHash: The paper connects MinHash to EVT via the Fisher-Tippett-Gnedenko theorem, stating that the minimum of n i.i.d. uniform(0,1) variables converges to exponential/Gumbel. However, MinHash minima are NOT the minimum of n i.i.d. uniform variables - they are the minimum of shingles' hash values, and shingles are overlapping (not independent) for text data. The paper acknowledges this in Discussion ('Dependence' section) but the Methods section doesn't clearly qualify the EVT assumption. This could mislead readers about the theoretical justification.
  Action: Clarify the EVT theoretical justification: (1) In the Methods section, add a clear statement that the Gumbel convergence assumes i.i.d. uniform hash values, which may not hold for MinHash of text due to overlapping shingles, (2) Move some of the Discussion points (dependence, small sample size) to the Methods section as 'Assumptions and Limitations' of the EVT approach, (3) Consider adding a simple simulation study: generate i.i.d. uniform data (truly random shingles), apply MinHash, and verify that EVT DOES work in this ideal case - this would isolate the dependence issue.
- [MINOR] (scope) Narrow evaluation scope: The evaluation is limited to short text documents (10-100 character-3 shingles) and MD5 hash function. The paper doesn't explore: (a) longer documents where EVT might work better, (b) different hash functions (e.g., MurmurHash which is more commonly used for MinHash), (c) different shingle types (word shingles, k-shingles with k≠3), (d) comparison with other uncertainty quantification methods beyond bootstrap.
  Action: Expand the evaluation scope: (1) Test on longer documents (100+ shingles) to see if EVT works better asymptotically, (2) Test with MurmurHash or other standard MinHash hash functions, (3) Compare with alternative uncertainty quantification methods: analytical variance bounds (based on binomial model for matching hashes), Bayesian approach, or jackknife. Even if EVT doesn't work, a comparison with multiple methods would strengthen the paper.
- [MINOR] (methodology) Bootstrap computational cost comparison is misleading: The paper compares EVT-MinHash (O(k) operations) with Bootstrap (O(B×k) operations, where B=1000). However, the bootstrap implementation in the paper is buggy (see major critique above), and the cost comparison uses the buggy implementation's timing. Additionally, the paper doesn't consider optimized bootstrap implementations or alternative resampling strategies that could reduce computational cost.
  Action: Provide a fair computational cost comparison: (1) Fix the bootstrap implementation first, (2) Time the corrected bootstrap implementation, (3) Consider that bootstrap can be optimized: (a) use vectorized operations, (b) parallelize resamples, (c) use smaller B if coverage is acceptable. (4) Compare not just wall-clock time but also memory usage and scalability. The conclusion should be: 'Even with optimizations, bootstrap costs O(B×k) vs O(k) for EVT, BUT EVT doesn't provide accurate CIs for MinHash, so the cost advantage is moot.'
</reviewer_feedback>



<task>
IMPORTANT: Your ONLY output is the revised hypothesis text. Do NOT run code, produce artifacts,
fix bugs, or attempt to address the evidence yourself — the next iteration of the invention loop
will generate fresh artifacts based on your revised hypothesis. Reflect and rewrite; nothing else.

Do NOT generate a completely new hypothesis. Take the current hypothesis and REVISE it
to incorporate new evidence. Keep the core idea — refine, narrow, or strengthen it.

1. Does the evidence support the hypothesis? Narrow or broaden scope as needed.
2. Which claims now have strong evidence? Which are still unsupported?
3. Should the hypothesis become more specific based on what we've learned?
4. If reviewer feedback is provided, address the critiques directly.

STABILITY IS OK: If progress is good and evidence supports the current direction, keep the
hypothesis similar or identical. Only make substantive changes when evidence clearly calls for
them — e.g., contradictory results, fundamental reviewer critiques, or findings that refine scope.

You must also classify two kinds of edges in the research trace:

(A) The H↔H edge — how does this revised hypothesis relate to the previous one?
    Set `relation_type` (Moulines's structuralist typology) to one of:
    - "evolution": refining specialised claims, same conceptual frame
    - "embedding": previous hypothesis is now a special case of a broader frame
    - "replacement": rejecting the previous frame entirely (Kuhnian shift)
    Set `relation_rationale` to a brief justification (≤120 chars).

(B) The A↔A edges — for each artifact created THIS iteration, classify each of its
    `in_dependencies` (predecessor → dependent) using MultiCite's citation-function
    typology (Lauscher et al., NAACL 2022) — emit one entry in `artifact_relations`
    per (predecessor, dependent) pair. Predecessors are ALWAYS artifacts from EARLIER
    iterations — artifacts within one iteration run in parallel and cannot depend on
    each other, so never emit a relation between two same-iteration artifacts (it
    will be dropped):
    - "background": predecessor is treated as background context
    - "motivation": predecessor motivated this artifact's research
    - "uses": this artifact uses the predecessor's data, method, or output
    - "extends": this artifact extends the predecessor
    - "similarities": this artifact's results agree with the predecessor's
    - "differences": this artifact's results disagree with the predecessor's
    Each `relation_rationale` must be ≤120 characters.

Output the COMPLETE revised hypothesis (with the H↔H relation fields) AND the full
list of A↔A `artifact_relations` for this iteration's new artifacts.
</task><user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>

---

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/upd_hypo/upd_hypo/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "ArtifactRelation": {
      "description": "One typed A\u2194A edge between a dependent artifact and one of its in_dependencies.\n\nMultiCite citation-function typology (Lauscher et al., NAACL 2022),\nreduced to 6 plain-English types.",
      "properties": {
        "from_id": {
          "description": "ID of the predecessor artifact (the one being depended on)",
          "title": "From Id",
          "type": "string"
        },
        "to_id": {
          "description": "ID of the dependent artifact (the new artifact this iteration)",
          "title": "To Id",
          "type": "string"
        },
        "relation_type": {
          "description": "MultiCite citation-function type for the predecessor\u2192dependent edge: 'background' \u2014 predecessor is treated as background context; 'motivation' \u2014 predecessor motivated this artifact's research; 'uses' \u2014 this artifact uses the predecessor's data, method, or output; 'extends' \u2014 this artifact extends the predecessor; 'similarities' \u2014 this artifact's results agree with the predecessor's; 'differences' \u2014 this artifact's results disagree with the predecessor's.",
          "enum": [
            "background",
            "motivation",
            "uses",
            "extends",
            "similarities",
            "differences"
          ],
          "title": "Relation Type",
          "type": "string"
        },
        "relation_rationale": {
          "description": "Brief rationale for this relation type (one short line, max 120 characters).",
          "maxLength": 120,
          "title": "Relation Rationale",
          "type": "string"
        }
      },
      "required": [
        "from_id",
        "to_id",
        "relation_type",
        "relation_rationale"
      ],
      "title": "ArtifactRelation",
      "type": "object"
    }
  },
  "description": "Revised hypothesis after reviewing iteration results.\n\nOutput matches the hypothesis dict structure so it can replace the\noriginal hypothesis in subsequent iterations.",
  "properties": {
    "title": {
      "description": "Revised hypothesis title (may be unchanged if still accurate)",
      "title": "Title",
      "type": "string"
    },
    "hypothesis": {
      "description": "Revised hypothesis statement \u2014 what we now believe based on evidence",
      "title": "Hypothesis",
      "type": "string"
    },
    "relation_rationale": {
      "description": "Brief rationale for the H\u2194H revision type (one short line, max 120 characters).",
      "maxLength": 120,
      "title": "Relation Rationale",
      "type": "string"
    },
    "confidence_delta": {
      "description": "How confidence changed: 'increased', 'decreased', or 'unchanged'",
      "title": "Confidence Delta",
      "type": "string"
    },
    "key_changes": {
      "description": "Bullet list of specific changes made to the hypothesis",
      "items": {
        "type": "string"
      },
      "title": "Key Changes",
      "type": "array"
    },
    "relation_type": {
      "description": "Moulines's structuralist typology of this hypothesis revision: 'evolution' \u2014 refining specialised claims while keeping the same conceptual frame; 'embedding' \u2014 the previous hypothesis is now a special case of a broader frame; 'replacement' \u2014 rejecting the previous frame entirely (incommensurable, Kuhnian revolution).",
      "enum": [
        "evolution",
        "embedding",
        "replacement"
      ],
      "title": "Relation Type",
      "type": "string"
    },
    "artifact_relations": {
      "description": "Typed A\u2194A edges for this iteration's new artifacts. Emit one entry per (predecessor \u2192 dependent) edge for every in_dependency on each artifact produced this iteration.",
      "items": {
        "$ref": "#/$defs/ArtifactRelation"
      },
      "title": "Artifact Relations",
      "type": "array"
    }
  },
  "required": [
    "title",
    "hypothesis",
    "relation_rationale",
    "confidence_delta",
    "key_changes",
    "relation_type"
  ],
  "title": "RevisedHypothesis",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_1/upd_hypo/upd_hypo/.sdk_openhands_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-06-22 05:42:04 UTC

```
Build and evaluate a simple MinHash near-duplicate detector for short text documents.
```
