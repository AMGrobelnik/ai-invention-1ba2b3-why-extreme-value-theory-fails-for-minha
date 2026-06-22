# upd_hypo — test_idea

> Phase: `invention_loop` · round 2 · `upd_hypo`
> Run: `run_EqcgJR2naF4b` — Why Extreme Value Theory Fails for MinHash Confidence Intervals on Short Text (and What to Do Instead)
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `upd_hypo` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 07:16:07 UTC

````
<current_hypothesis>
The hypothesis as it stands. Revise it based on the evidence below.

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

--- Item 4 ---
id: art_N6UOTzkLgQlx
type: experiment
in_dependencies:
- id: art_qBzTyA1I1xtt
  label: dataset
- id: art_T61Vq8hqL41x
  label: research
title: Bootstrap CI Fix for MinHash on Short Text Data - Experiment Results
summary: >-
  Implemented and executed experiment to fix bootstrap CI implementation for MinHash on short text documents from tweet_eval
  and ag_news datasets. The bug was that resampling was done on MinHash signatures instead of shingles. Correct implementation
  resamples shingles, recomputes signatures, and estimates Jaccard similarity. Results: correct method achieved 86% coverage
  (target 95%), incorrect method achieved 0% coverage (confirming the bug). Average CI width on real data: 0.059. CI contains
  point estimate rate: 100%. Experiment demonstrates the fix is correct in principle; coverage approaches target with larger
  B.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json

--- Item 5 ---
id: art_7G0p4_rkOjnY
type: experiment
in_dependencies:
- id: art_qBzTyA1I1xtt
  label: dataset
- id: art_T61Vq8hqL41x
  label: validates
title: >-
  Alternative Uncertainty Quantification for MinHash: Binomial and Bayesian Methods
summary: "This experiment implements and evaluates two practical alternatives to EVT-MinHash for uncertainty quantification:\
  \ (1) analytical binomial confidence intervals (Clopper-Pearson exact and Wilson score) based on matching hash counts, and\
  \ (2) Bayesian approach with Beta prior informed by document length. The methods are compared against a bootstrap baseline\
  \ on three short text datasets from HuggingFace Hub (tweet_eval sentiment/emoji with ~11-19 words, ag_news headlines with\
  \ ~39 words).\n\nKey findings:\n1. Coverage Probability: All methods achieved 100% coverage on the test dataset with 100\
  \ pairs and 25 runs, indicating conservative intervals. The nominal target was 95%, suggesting the intervals may be too\
  \ wide for practical use.\n\n2. Computational Cost: Analytical methods are significantly faster than bootstrap:\n   - Clopper-Pearson:\
  \ 84 microseconds per CI\n   - Wilson score: 41 microseconds per CI  \n   - Bayesian (Beta posterior): 395 microseconds\
  \ per CI\n   - Bootstrap (B=100): ~2000+ microseconds per CI\n   The analytical methods provide 10-50x speedup over bootstrap.\n\
  \n3. Interval Width: Binomial methods (Clopper-Pearson, Wilson) produce wider intervals for small k values, while Bayesian\
  \ methods with length-informed priors adapt interval width based on document characteristics.\n\n4. Practical Implications:\
  \ For near-duplicate detection in short texts, the binomial CI approach offers a computationally efficient alternative to\
  \ bootstrap with similar statistical properties. The Bayesian approach provides a principled way to incorporate document\
  \ length information into uncertainty estimates.\n\nLimitations: The experiment used reduced parameters (100 pairs, 25 runs,\
  \ k=64/128) due to computational constraints. Full evaluation with 500 pairs and 50 runs would provide more precise coverage\
  \ estimates. The 100% coverage suggests either conservative intervals or insufficient sample diversity."
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_art/gen_art_experiment_2
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json

--- Item 6 ---
id: art_UUINZzLEnsrT
type: evaluation
in_dependencies:
- id: art_SiAaMxtNmflp
  label: baseline
- id: art_qBzTyA1I1xtt
  label: data
title: Comprehensive Evaluation of UQ Methods for MinHash Jaccard Estimates
summary: >-
  This evaluation compares 5 uncertainty quantification methods (EVT-Gumbel, EVT-Weibull, Corrected Bootstrap, Analytical
  Binomial, Bayesian) for MinHash Jaccard estimates on short text (10-100 shingles). The evaluation processed 3000 document
  pairs from 3 datasets (tweets, SMS, headlines). Key findings: (1) Analytical Binomial and Bayesian methods achieve 96.5%
  and 94.8% coverage respectively (within 2% of 95% target), (2) EVT methods achieve acceptable coverage but are theoretically
  unjustified for short text, (3) Corrected Bootstrap only achieves 75.5% coverage with 1000 resamples. Output includes eval_out.json
  (validated against exp_eval_sol_out schema), 6 figures (coverage probability, CI width, coverage by regime, bias distribution,
  CI width vs shingle count, computational cost), 3 tables (coverage, CI width stats, bias/RMSE), and practitioner guidelines.
  The evaluation supports the hypothesis that practitioners should use analytical binomial or Bayesian methods for MinHash
  uncertainty quantification on short text.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_art/gen_art_evaluation_1
out_expected_files:
- eval.py
- full_eval_out.json
- mini_eval_out.json
- preview_eval_out.json
</all_artifacts>

<new_artifacts_this_iteration>
These 3 artifacts were created THIS iteration.

id: art_N6UOTzkLgQlx
type: experiment
in_dependencies:
- id: art_qBzTyA1I1xtt
  label: dataset
- id: art_T61Vq8hqL41x
  label: research
title: Bootstrap CI Fix for MinHash on Short Text Data - Experiment Results
summary: >-
  Implemented and executed experiment to fix bootstrap CI implementation for MinHash on short text documents from tweet_eval
  and ag_news datasets. The bug was that resampling was done on MinHash signatures instead of shingles. Correct implementation
  resamples shingles, recomputes signatures, and estimates Jaccard similarity. Results: correct method achieved 86% coverage
  (target 95%), incorrect method achieved 0% coverage (confirming the bug). Average CI width on real data: 0.059. CI contains
  point estimate rate: 100%. Experiment demonstrates the fix is correct in principle; coverage approaches target with larger
  B.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json

id: art_7G0p4_rkOjnY
type: experiment
in_dependencies:
- id: art_qBzTyA1I1xtt
  label: dataset
- id: art_T61Vq8hqL41x
  label: validates
title: >-
  Alternative Uncertainty Quantification for MinHash: Binomial and Bayesian Methods
summary: "This experiment implements and evaluates two practical alternatives to EVT-MinHash for uncertainty quantification:\
  \ (1) analytical binomial confidence intervals (Clopper-Pearson exact and Wilson score) based on matching hash counts, and\
  \ (2) Bayesian approach with Beta prior informed by document length. The methods are compared against a bootstrap baseline\
  \ on three short text datasets from HuggingFace Hub (tweet_eval sentiment/emoji with ~11-19 words, ag_news headlines with\
  \ ~39 words).\n\nKey findings:\n1. Coverage Probability: All methods achieved 100% coverage on the test dataset with 100\
  \ pairs and 25 runs, indicating conservative intervals. The nominal target was 95%, suggesting the intervals may be too\
  \ wide for practical use.\n\n2. Computational Cost: Analytical methods are significantly faster than bootstrap:\n   - Clopper-Pearson:\
  \ 84 microseconds per CI\n   - Wilson score: 41 microseconds per CI  \n   - Bayesian (Beta posterior): 395 microseconds\
  \ per CI\n   - Bootstrap (B=100): ~2000+ microseconds per CI\n   The analytical methods provide 10-50x speedup over bootstrap.\n\
  \n3. Interval Width: Binomial methods (Clopper-Pearson, Wilson) produce wider intervals for small k values, while Bayesian\
  \ methods with length-informed priors adapt interval width based on document characteristics.\n\n4. Practical Implications:\
  \ For near-duplicate detection in short texts, the binomial CI approach offers a computationally efficient alternative to\
  \ bootstrap with similar statistical properties. The Bayesian approach provides a principled way to incorporate document\
  \ length information into uncertainty estimates.\n\nLimitations: The experiment used reduced parameters (100 pairs, 25 runs,\
  \ k=64/128) due to computational constraints. Full evaluation with 500 pairs and 50 runs would provide more precise coverage\
  \ estimates. The 100% coverage suggests either conservative intervals or insufficient sample diversity."
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_art/gen_art_experiment_2
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json

id: art_UUINZzLEnsrT
type: evaluation
in_dependencies:
- id: art_SiAaMxtNmflp
  label: baseline
- id: art_qBzTyA1I1xtt
  label: data
title: Comprehensive Evaluation of UQ Methods for MinHash Jaccard Estimates
summary: >-
  This evaluation compares 5 uncertainty quantification methods (EVT-Gumbel, EVT-Weibull, Corrected Bootstrap, Analytical
  Binomial, Bayesian) for MinHash Jaccard estimates on short text (10-100 shingles). The evaluation processed 3000 document
  pairs from 3 datasets (tweets, SMS, headlines). Key findings: (1) Analytical Binomial and Bayesian methods achieve 96.5%
  and 94.8% coverage respectively (within 2% of 95% target), (2) EVT methods achieve acceptable coverage but are theoretically
  unjustified for short text, (3) Corrected Bootstrap only achieves 75.5% coverage with 1000 resamples. Output includes eval_out.json
  (validated against exp_eval_sol_out schema), 6 figures (coverage probability, CI width, coverage by regime, bias distribution,
  CI width vs shingle count, computational cost), 3 tables (coverage, CI width stats, bias/RMSE), and practitioner guidelines.
  The evaluation supports the hypothesis that practitioners should use analytical binomial or Bayesian methods for MinHash
  uncertainty quantification on short text.
workspace_path: >-
  /ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_art/gen_art_evaluation_1
out_expected_files:
- eval.py
- full_eval_out.json
- mini_eval_out.json
- preview_eval_out.json
</new_artifacts_this_iteration>

<current_paper>
The paper draft from this iteration — represents the current state of the research story.

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

2. **Empirical Verification**: We implement MinHash with 128 hash functions using MD5 and empirically test whether MinHash signature minima follow Gumbel or Weibull distributions for short text documents (10-100 character shingles). We use real-world datasets from HuggingFace Hub (tweet_eval, ag_news) with 39,951 documents [ARTIFACT:art_qBzTyA1I1xtt].

3. **Distribution Fit Analysis**: We perform Kolmogorov-Smirnov (KS) tests and Anderson-Darling (AD) tests to evaluate Gumbel and Weibull distribution fits across three datasets and multiple shingle count ranges (10-30, 30-50, 50-100).

4. **Negative Results on EVT**: We report that neither Gumbel nor Weibull distributions provide adequate fit for MinHash minima from short text documents, with all KS test p-values < 10^{-20}. This suggests that the asymptotic EVT approximation does not hold in the small-sample regime due to finite-sample bias, dependence between overlapping shingles, and hash function discretization.

5. **Alternative UQ Methods**: We implement and evaluate two practical alternatives to EVT-MinHash: (a) analytical binomial confidence intervals (Clopper-Pearson exact and Wilson score) based on matching hash counts, and (b) Bayesian approach with Beta prior informed by document length [ARTIFACT:art_7G0p4_rkOjnY].

6. **Comprehensive Evaluation**: We compare five uncertainty quantification methods (EVT-Gumbel, EVT-Weibull, Corrected Bootstrap, Analytical Binomial, Bayesian) on 3,000 document pairs from real-world datasets. Results show that Analytical Binomial and Bayesian methods achieve 96.5% and 94.8% coverage (within 2% of 95% target), while Corrected Bootstrap only achieves 75.5% coverage [ARTIFACT:art_UUINZzLEnsrT].

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

We evaluate on real-world datasets from HuggingFace Hub: Tweet Eval Sentiment (10K tweets, avg 68.5 shingles), Tweet Eval Emoji (10K tweets, avg 53.2 shingles), AG News (19,951 headlines, avg 156.8 shingles). Total: 39,951 documents [ARTIFACT:art_qBzTyA1I1xtt].

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
</current_paper>

<reviewer_feedback>
Feedback from the paper reviewer this iteration.

- [MAJOR] (novelty) The main contribution is negative results (EVT doesn't work) plus straightforward applications of known methods (binomial CIs, Bayesian CIs). There is no new method, no new theoretical insight, and no surprising finding. The paper reads more like an experimental report than a research contribution.
  Action: To raise the paper to top-tier quality: (1) Derive a finite-sample correction for EVT that accounts for small k (number of shingles) - the exact distribution of minima is known for i.i.d. uniform variables, use that instead of asymptotic EVT; (2) Propose a novel UQ method that combines analytical and bootstrap approaches (e.g., analytical variance bound with bootstrap bias correction); (3) Provide a theoretical analysis of the dependence bias in MinHash - derive how overlapping shingles affect the distribution of matching hashes. Any one of these would substantially increase the contribution.
- [MAJOR] (evidence) The paper claims 'all KS test p-values < 10^{-20}' but doesn't show the actual test statistics or p-values in the main text. Table 1 is referenced but not included in the paper. This makes it difficult to verify the strength of evidence against EVT. Additionally, the coverage results (96.5%, 94.8%) are reported without confidence intervals - we don't know if these are significantly different from 95%.
  Action: 1. Include Table 1 (or a summary table) in the main text with KS statistics and p-values for all conditions. 2. Compute confidence intervals for the coverage probabilities. With 3,000 pairs, the standard error of coverage is sqrt(0.95*0.05/3000) ≈ 0.004, so the 96.5% and 94.8% coverage rates are within 2-3 SE of 95% - they may not be significantly different. Report SE or CI for coverage estimates.
- [MAJOR] (methodology) The analytical binomial model assumes that the k MinHash hashes are independent Bernoulli trials with probability J of matching. However, this assumption is questionable: (1) The k hash functions are applied to the same sets, so the matches are not independent, (2) The shingles within a document are overlapping, creating dependence, (3) The binomial model ignores the fact that MinHash estimates J with bias that depends on set sizes. The paper should acknowledge these limitations and ideally provide evidence that the binomial model is adequate despite violations of independence.
  Action: 1. Add a simulation study: generate data where true J is known (e.g., synthetic sets with controlled overlap) and verify that binomial CIs have correct coverage. 2. Compare binomial CIs with CIs from simulation-based methods (e.g., repeated MinHash with different hash seeds) to check if binomial CIs are properly calibrated. 3. Add a discussion of the independence assumption and its violations. If the binomial model is found to be inadequate, propose corrections.
- [MINOR] (scope) The evaluation is limited to short text documents (10-100 shingles) and k=128 hash functions. The paper doesn't explore: (1) Longer documents where EVT might work better (asymptotically), (2) Different numbers of hash functions (k=32, 64, 256, 512) to see how UQ methods scale, (3) Different shingle types (word shingles, k-shingles with k≠3), (4) Different hash functions (MurmurHash is more common than MD5 for MinHash).
  Action: Expand the evaluation: (1) Test on longer documents (100+ shingles) to see if EVT asymptotically approaches correctness. (2) Test with different k values to show how CI width and coverage depend on k. (3) Test with word shingles in addition to character shingles. (4) Use MurmurHash or other standard MinHash hash functions. Even if resources are limited, adding one or two of these would strengthen the paper.
- [MINOR] (clarity) The connection between MinHash and EVT is not clearly explained. The paper states that MinHash signature values are minima of hash values, which converge to Gumbel under EVT. However, it doesn't explain: (1) What is the 'block size' for EVT? (MinHash uses k independent hashes, not blocks of data), (2) How are Gumbel parameters estimated from MinHash signatures? (the paper mentions MLE but doesn't show the procedure), (3) How does the Gumbel distribution relate to Jaccard similarity? (the paper says 'delta method can propagate uncertainty' but doesn't derive the relationship).
  Action: Add a clear derivation: (1) Show that MinHash signature for a single hash function is min(h(s1), h(s2), ..., h(sn)) where s1...sn are shingles. (2) If hash values are uniform(0,1) and independent, the minimum has CDF = 1 - (1 - x)^n ≈ 1 - exp(-n*x) for large n, which is exponential, not Gumbel. (3) The Gumbel distribution arises when we consider the minimum across multiple documents or across blocks - clarify this connection. (4) Derive or cite the relationship between Gumbel parameters and Jaccard similarity.
- [MINOR] (rigor) The paper evaluates UQ methods based on coverage probability and CI width, but doesn't consider other important metrics: (1) Mean squared error of the UQ method (bias-variance tradeoff), (2) Computational cost as a function of k and B (number of bootstrap samples), (3) Sensitivity to the choice of prior (for Bayesian method), (4) Performance on edge cases (J≈0 or J≈1).
  Action: Expand the evaluation metrics: (1) Add MSE of the UQ method (not just the Jaccard estimate). (2) Show computational cost scaling with k and B. (3) For Bayesian method, test different priors (uniform, length-informed, Jeffreys) and show sensitivity. (4) Evaluate on extreme similarities (J<0.1 or J>0.9) where UQ is most challenging.
- [MINOR] (methodology) The bootstrap implementation resamples shingles and recomputes signatures. This is computationally expensive (O(B * k * L) where L is document length). The paper could consider more efficient bootstrap variants: (1) Parametric bootstrap: resample from the empirical distribution of hash values, (2) Weighted bootstrap: use weights on shingles instead of resampling, (3) Bag of little bootstraps (BLB) for reduced computation.
  Action: Compare the corrected bootstrap with more efficient bootstrap variants. If the corrected bootstrap is too slow for practical use, recommend an efficient alternative. Alternatively, provide guidelines on choosing B (number of bootstrap samples) to achieve target coverage with minimal computation.
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

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/upd_hypo/upd_hypo/.sdk_openhands_agent_struct_out.json`

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

IMPORTANT: This task is NOT complete until you Write `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/upd_hypo/upd_hypo/.sdk_openhands_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-06-22 07:16:07 UTC

```
Build and evaluate a simple MinHash near-duplicate detector for short text documents.
```

### [3] SYSTEM-USER prompt · 2026-06-22 07:17:48 UTC

```
<validation-feedback>
Attempt 1 failed validation.

Schema validation found 3 problems — fix ALL of them at once:
  - at `<root>`: 'relation_rationale' is a required property
  - at `<root>`: 'confidence_delta' is a required property
  - at `<root>`: 'key_changes' is a required property
Every required field must be present and every field type must match the schema.

Please use the Write tool to overwrite `.sdk_openhands_agent_struct_out.json` with corrected JSON. Do not invent new fields; match the schema you were given.
</validation-feedback>
```
