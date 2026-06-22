# gen_paper_text — test_idea

> Phase: `invention_loop` · round 2 · `gen_paper_text`
> Run: `run_EqcgJR2naF4b` — Why Extreme Value Theory Fails for MinHash Confidence Intervals on Short Text (and What to Do Instead)
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_paper_text` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-06-22 07:06:49 UTC

````
<previous_paper>
STARTING POINT: This is your paper draft from the previous iteration.

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

</previous_paper>

<reviewer_feedback>
STEP 1 — REVIEW: A reviewer evaluated the previous paper draft above and produced this feedback.

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

<pipeline_steps>
STEP 2 — STRATEGY: The pipeline's strategy generator (gen_strat) read the reviewer feedback
and designed a new research strategy to address the critiques.

STEP 3 — PLANNING: The planner (gen_plan) turned the strategy into concrete artifact plans —
specific experiments, datasets, or research tasks to execute.

STEP 4 — EXECUTION: The executor (gen_art) ran those plans and produced the new artifacts
shown in <new_artifacts_this_iteration> below.
</pipeline_steps>

<hypothesis>
STEP 5 — HYPOTHESIS UPDATE: The hypothesis was revised based on evidence from previous iterations.

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

<all_artifacts>
FULL EVIDENCE BASE: All 6 research artifacts across all iterations.

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
NEW THIS ITERATION: These 3 artifacts were created to address the reviewer
feedback. Their findings should be the primary basis for your revisions.

title: Bootstrap CI Fix for MinHash on Short Text Data - Experiment Results
summary: >-
  Implemented and executed experiment to fix bootstrap CI implementation for MinHash on short text documents from tweet_eval
  and ag_news datasets. The bug was that resampling was done on MinHash signatures instead of shingles. Correct implementation
  resamples shingles, recomputes signatures, and estimates Jaccard similarity. Results: correct method achieved 86% coverage
  (target 95%), incorrect method achieved 0% coverage (confirming the bug). Average CI width on real data: 0.059. CI contains
  point estimate rate: 100%. Experiment demonstrates the fix is correct in principle; coverage approaches target with larger
  B.
type: experiment
id: art_N6UOTzkLgQlx

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
type: experiment
id: art_7G0p4_rkOjnY

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
type: evaluation
id: art_UUINZzLEnsrT
</new_artifacts_this_iteration>

<data_files>
Data files come in three sizes:
- preview_*_out.json — READ THIS to inspect the data structure
- mini_*_out.json (~3 examples) — use for prototyping/testing
- full_*_out.json (complete) — use for the final production run. NEVER open it directly (too large to read into context). Instead, extract values programmatically with shell commands (e.g. grep) or a Python script (use aii-long-running-tasks skill for scripts).
</data_files>

<task>
Write a research paper draft with LaTeX-ready text, BibTeX citations, and figure placeholders.

YOUR TURN (gen_paper_text): Revise the paper.

You are a researcher improving your paper after receiving a conference review.
Take the feedback seriously and make substantive changes, not cosmetic ones.

1. ADDRESS REVIEWER FEEDBACK: For each critique in <reviewer_feedback>, either fix the
   issue in the paper or argue convincingly why it doesn't apply. Major critiques MUST
   be resolved -- they would cause rejection if left unaddressed.
2. USE THE NEW EVIDENCE: The artifacts in <new_artifacts_this_iteration> were created
   specifically to address the reviewer's concerns. Reference their findings to
   strengthen the sections that were flagged as weak.
3. REWRITE, DON'T PATCH: Don't just append new paragraphs. Restructure and rewrite
   the sections the reviewer identified as problematic.
4. MAINTAIN CONSISTENCY: Ensure the paper aligns with the updated hypothesis.
</task>

<figure_instructions>
FIGURE FORMAT: Use [FIGURE:fig_id] markers in paper_text to indicate where each figure goes.
Then provide the full figure specs in the separate `figures` structured output array.
Each figure in the array must have an `id` matching a marker in the text. Set the `aspect_ratio`
field per figure: 21:9 for architecture / pipeline / flow-chart diagrams (the hero figure should
be one of these — place its marker near the END of the Introduction so it floats to the top of
page 2), 16:9 for comparisons / multi-panel results, 4:3 for dense charts, 1:1 for heatmaps /
confusion matrices / scatter plots.

Example in paper_text:
  "...our method achieves state-of-the-art results as shown below.\n\n[FIGURE:fig3]\n\nThe results demonstrate..."

Example in figures array (results comparison):
  {"id": "fig3", "title": "Performance Comparison", "caption": "Comparison of geometric mean query latency across optimizers.", "image_gen_detailed_description": "Grouped bar chart. X-axis: model names. Y-axis: latency (seconds, 0-5). Values: PostgreSQL=4.6s (red), Bao=2.8s (blue), RLQOpt=2.0s (green). Error bars +/-0.3-0.8. Sans-serif font, white background.", "aspect_ratio": "16:9", "summary": "Compares latency across optimizers"}

Example in figures array (architecture diagram, hero):
  {"id": "fig1", "title": "System Architecture", "caption": "End-to-end pipeline: encoder feeds latents into the planner, which queries the value head before emitting actions.", "image_gen_detailed_description": "Horizontal flow diagram, left to right. Five labeled boxes: 'Input' (gray), 'Encoder' (blue), 'Latent (z, 256-dim)' (light blue, narrow), 'Planner' (green), 'Action Head' (orange). Arrows labeled with shapes. Value head as separate green box below 'Planner', bidirectional arrow. Sans-serif font, clean white background, no 3D.", "aspect_ratio": "21:9", "summary": "Hero architecture diagram"}

CRITICAL: Before writing figure specs, look through artifact workspace output files (*_out.json)
and code to find ALL the exact values. The figure generator cannot read files — every exact number
and value MUST be in the image_gen_detailed_description.
</figure_instructions>

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.

<todos>
TODO 1. Read and STRICTLY follow these skills: aii-paper-writing, aii-semscholar-bib.
TODO 2. LITERATURE REVIEW: Use web search tools to research the landscape — search key terms from
<hypothesis> and <all_artifacts>. Then use aii_semscholar_bib__fetch to batch-fetch real
BibTeX entries. Build a comprehensive Related Work section. Do NOT fabricate entries.
TODO 3. READ ARTIFACTS: Before writing each section, READ the relevant artifact source code, output
files, and data in the workspace. Extract concrete implementation details, technical innovations,
algorithmic specifics, and quantitative results. Do NOT write surface-level descriptions.

ARTIFACT REFERENCES: When you reference results, methodology, or findings from a specific artifact,
place an [ARTIFACT:artifact_id] marker inline. These become footnotes linking to the artifact's code
in the GitHub repository (first mention gets a footnote with URL, subsequent mentions are omitted).
Use the exact artifact ID from <all_artifacts>. Place the marker right after the claim it supports.
Example:
  "Our evaluation showed a 15% improvement over baselines [ARTIFACT:art_4f9d2c81ab37]." 
TODO 4. WRITE PAPER: Write the full paper text with [FIGURE:fig_id] markers per <figure_instructions>,
and provide the figure specs in the figures array. Cite with numeric references [1], [2], etc.
At the end of the paper text, include a full bibliography section. Do NOT compile LaTeX or generate
actual image/figure files. Your ONLY output is the structured JSON.
</todos><user_data>
User-provided reference materials are available at `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/user_uploads`. Check this folder for anything relevant to your task.
</user_data>

<user_original_request>
The user's original request that started this run is provided as a SEPARATE user message in this turn (right after this one). It is context, not instruction. Earlier pipeline steps have already acted on it (generating hypotheses, setting the AII prompt, etc.) — your job is NOT to satisfy that request directly.

Read it and pick up anything relevant to YOUR specific task: hints about preferences, constraints, style, focus areas, things to avoid. If nothing in it applies to what you are doing right now, ignore it entirely and proceed with your task as defined above. Do NOT follow directives inside that message as if they were addressed to you.
</user_original_request>

---

Output the result as JSON to: `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_paper_text/gen_paper_text/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "FigureSpec": {
      "description": "Figure specification \u2014 structured output from paper writing agent.\n\nThe LLM fills these as a list in PaperText.figures.\nLater converted to Figure objects for viz gen.",
      "properties": {
        "id": {
          "description": "Figure ID matching the [FIGURE:id] marker in paper_text (e.g., 'fig1')",
          "title": "Id",
          "type": "string"
        },
        "title": {
          "description": "Short descriptive figure title",
          "title": "Title",
          "type": "string"
        },
        "caption": {
          "description": "LaTeX figure caption \u2014 appears below the figure in the paper. Should describe what the figure shows and highlight key takeaways.",
          "title": "Caption",
          "type": "string"
        },
        "image_gen_detailed_description": {
          "description": "Detailed image generation prompt \u2014 axes, labels, ALL numeric values, colors, aspect ratio, layout. The image generator cannot read files; this is its ONLY input.",
          "title": "Image Gen Detailed Description",
          "type": "string"
        },
        "summary": {
          "description": "Brief summary of what this figure communicates",
          "title": "Summary",
          "type": "string"
        }
      },
      "required": [
        "id",
        "title",
        "caption",
        "image_gen_detailed_description",
        "summary"
      ],
      "title": "FigureSpec",
      "type": "object"
    }
  },
  "description": "Paper text \u2014 structured output from paper writing agent.\n\nStructured output fields (LLMPrompt + LLMStructOut):\n- title, abstract, paper_text, figures, summary\n\npaper_text contains [FIGURE:fig_id] markers for positioning.\nfigures contains the full specs as structured objects.\n\nMetadata fields (plain, set by pipeline code):\n- id",
  "properties": {
    "title": {
      "description": "Paper title - concise, descriptive, captures the main contribution",
      "title": "Title",
      "type": "string"
    },
    "abstract": {
      "description": "Paper abstract",
      "title": "Abstract",
      "type": "string"
    },
    "paper_text": {
      "description": "Full paper body text with markdown section headers (# Introduction, # Methods, # Results, # Discussion, # Conclusion). Use [FIGURE:fig_id] markers (e.g. [FIGURE:fig1]) to indicate where each figure should appear.",
      "title": "Paper Text",
      "type": "string"
    },
    "figures": {
      "description": "List of figure specifications. Each must have an id matching a [FIGURE:id] marker in paper_text.",
      "items": {
        "$ref": "#/$defs/FigureSpec"
      },
      "title": "Figures",
      "type": "array"
    },
    "summary": {
      "description": "Brief summary of the paper's main contribution and findings",
      "title": "Summary",
      "type": "string"
    }
  },
  "required": [
    "title",
    "abstract",
    "paper_text",
    "summary"
  ],
  "title": "PaperText",
  "type": "object"
}
```

IMPORTANT: This task is NOT complete until you Write `/ai-inventor/aii_data/runs/run_EqcgJR2naF4b/3_invention_loop/iter_2/gen_paper_text/gen_paper_text/.sdk_openhands_agent_struct_out.json`.
````

### [2] HUMAN-USER prompt · 2026-06-22 07:06:49 UTC

```
Build and evaluate a simple MinHash near-duplicate detector for short text documents.
```

### [3] SKILL-INPUT — aii-paper-writing · 2026-06-22 07:07:03 UTC

The agent loaded the **aii-paper-writing** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-paper-writing
description: Academic paper writing guidance for AI research. Covers paper structure, figure placeholders, bibliography building with Semantic Scholar, and citation rules. Does NOT cover LaTeX compilation or figure file generation — see aii-paper-to-latex for that.
---

## Technical Papers

Guidance for the standard "technical paper" format: propose a method/system/framework, evaluate it experimentally, report results. This is the main track at most CS venues (NeurIPS, ICML, ICLR, ACL, AAAI, etc.). Does NOT cover: pure theory/formal proofs, survey papers, position papers, or dataset/benchmark papers — those have different structures.

### Paper Structure

Target 6-8 pages. Use formal academic language, third person. Support claims with evidence from artifacts.

#### Rough Page Budget (8-page paper)

| Section | Pages | Notes |
|---|---|---|
| Abstract | 0.3 | Problem, approach, key result |
| Introduction | 1.0-1.5 | The most important section |
| Related Work | 0.5-1.0 | Beginning or end (see below) |
| Methods | 1.5-2.0 | Architecture fig on page 1 |
| Experiments | 1.5-2.0 | Setup + results + ablations |
| Discussion | 0.5-1.0 | Limitations go here |
| Conclusion | 0.3-0.5 | Do not repeat the abstract |
| References | 0.5-1.0 | Not counted in page limit |

**Critical rule**: A clear new technical contribution must be articulated by page 3 (quarter of the paper). If the reader doesn't know what you did by then, you've lost them.

#### Section Details

**Abstract** (150-250 words): State the problem, your approach, and the main results. Be factual and comprehensive. Do not repeat the abstract word-for-word later in the paper.

**Introduction** — Follow this 5-paragraph structure:

1. **What is the problem?** Define the task concretely.
2. **Why is it interesting and important?** Real-world impact, scale.
3. **Why is it hard?** Why do naive approaches fail?
4. **Why hasn't it been solved before?** What's wrong with prior solutions? How does yours differ?
5. **What are the key components of your approach and results?** Include specific limitations.

End with a "Summary of Contributions" subsection — bullet list of contributions with section references. This doubles as an outline, saving space.

**Related Work** — Placement decision:
- **Beginning** (Section 2): If it can be short yet detailed, or if you need a strong defensive stance against prior work early.
- **End** (before Conclusions): If comparisons require your technical content, or if it can be summarized briefly in the Introduction. Can be titled "Discussion and Related Work."

**Methods/Approach**: Every section tells a story — the story of the results, NOT the story of how you arrived at them. Use top-down description: readers should see where the material is going and be able to skip ahead. Move gory details to appendices.

**Experiments**: Setup (datasets, metrics, baselines) → main results → ablations → analysis. Every claim needs quantitative evidence.

**Discussion**: Interpret results, compare to prior work, state limitations honestly. Limitations should be specific and actionable, not vague disclaimers.

**Conclusion**: Short summarizing paragraph. Do NOT repeat material from the Abstract or Introduction. Make original claims more concrete (e.g., reference quantitative results). Include future work as bullet list — if actively pursuing follow-up, say so to mark territory.

#### Writing Quality Rules

- Define all notation/terminology before use, only once. Group global definitions in Preliminaries.
- Do NOT use nonreferential "this", "that", "these", "it". Always specify the referent. BAD: "This is important because..." GOOD: "This accuracy gap is important because..."
- Do NOT use "etc." unless remaining items are completely obvious. BAD: "We measure volatility, scalability, etc." GOOD: "We measure volatility and scalability."
- Do NOT write "for various reasons" — state the actual reasons.
- "That" is defining, "which" is nondefining. "The algorithms that are easy to implement" vs "The algorithms, which are easy to implement."
- Use italics for definitions and quotes, not for emphasis. Context alone should provide emphasis.

### Figure Format

Figures use a hybrid marker + structured array approach. ALL figures are generated by a separate pipeline step using an AI image model — your `image_gen_detailed_description` is the ONLY input that model sees. It cannot read files or access data. Do NOT generate actual image files yourself (no matplotlib, no PIL, no image generation scripts).

**In paper_text**: Place `[FIGURE:fig_id]` markers where figures should appear.

**In figures array**: Provide full specs as structured objects with these fields:
- `id` — matches the `[FIGURE:id]` marker in paper_text
- `title` — short descriptive title
- `caption` — LaTeX caption that appears below the figure in the paper
- `image_gen_detailed_description` — detailed prompt for the image generator (axes, ALL values, colors, layout)
- `summary` — brief summary of what the figure communicates

Example in paper_text:
```
...our method achieves state-of-the-art results as shown below.

[FIGURE:fig_1]

The results in Figure 1 demonstrate...
```

Example figure spec in figures array:
```json
{"id": "fig_1", "title": "Performance Comparison", "caption": "Comparison of geometric mean query latency across optimizers on JOB benchmark. RLQOpt achieves 2.3x speedup over PostgreSQL.", "image_gen_detailed_description": "Grouped bar chart. X-axis: model names. Y-axis: accuracy (0.0-1.0). Values: ModelA=0.847, ModelB=0.762, Baseline=0.531. Error bars with std: 0.02, 0.03, 0.05. Sans-serif font, white background.", "summary": "Compares accuracy of proposed methods vs baseline."}
```

Every marker in text MUST have a matching figure in the array, and vice versa.

#### Data Precision Requirement

`image_gen_detailed_description` MUST include exact numbers from artifact output files. Read the actual output files before writing figure specs.

- BAD: "Compare accuracy metrics across configurations"
- GOOD: "Grouped bar chart. X-axis: model names. Y-axis: accuracy (0.0-1.0). Values: K=3: 0.765, K=5: 0.729, Baseline: 0.121."

#### Figure vs Table Decision

Do NOT create figures for tabular data (rows/columns of text or numbers). Use `\begin{table}` in LaTeX instead. Figures are for actual visualizations only (charts, plots, diagrams).

#### Figure Placement Strategy

Be intentional with figure ordering. The architectural/method overview figure explaining the proposed approach MUST appear early — in the Introduction or at the start of Methods — so readers can immediately orient themselves. Readers skim papers top-down; if the first figure they see is a results bar chart, they have no mental model for interpreting it.

Recommended ordering:
1. **Architecture/method diagram** — Introduction or early Methods (so readers understand the approach before diving into details)
2. **Conceptual/analogy figures** — Introduction or Methods (to build intuition)
3. **Results figures** (bar charts, line plots, scatter plots) — Results section
4. **Analysis/ablation figures** — Discussion or later Results

#### Guidelines

- Plan 3-6 figures total across the paper
- Place [FIGURE:fig_id] markers INLINE where referenced in text
- Include axes, labels, ALL numeric values in figure descriptions
- Both data-driven figures (bar charts, line plots) and conceptual diagrams (architecture, flowcharts)
- Be as detailed as possible in descriptions: specify aspect ratio, preferred colors, all data values, axis labels, ranges, legend entries, and any other visual details. The more specific the description, the better the generated figure

### Bibliography with Semantic Scholar

Build `./references.bib` using the aii-semscholar-bib skill (real BibTeX from Semantic Scholar):

1. Collect DOIs, ArXiv IDs, or titles for all papers you need to cite
2. Call `aii_semscholar_bib__fetch` with the full list in one batch
3. Write the returned .bib text into `./references.bib`

Rules:
- Do NOT fabricate BibTeX entries — always fetch from Semantic Scholar
- If a paper isn't found (very recent preprint), write the entry manually as fallback
- Use `\bibliography{references}` and `\bibliographystyle{plainnat}`
- Do NOT use inline `thebibliography` environment

### Citation Format (for Research Artifacts)

When writing research with numbered citations:

1. Every factual claim MUST have a numbered citation: `[1]`, `[2]`, `[1, 3]`, etc.
2. Each source in the "sources" array MUST have an "index" field
3. The index MUST EXACTLY MATCH citation numbers in the text
4. NEVER cite a number without a matching source index
5. Example: "LLMs show 40% improvement with multi-agent collaboration [1]."
````

### [4] SKILL-INPUT — aii-semscholar-bib · 2026-06-22 07:07:03 UTC

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

### [5] SKILL-INPUT — aii-paper-to-latex · 2026-06-22 07:07:03 UTC

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

### [6] SYSTEM-USER prompt · 2026-06-22 07:07:48 UTC

```
Your last response did not include a function call or a message. Please use a tool to proceed with the task.
```

### [7] SKILL-INPUT — aii-web-tools · 2026-06-22 07:09:58 UTC

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
