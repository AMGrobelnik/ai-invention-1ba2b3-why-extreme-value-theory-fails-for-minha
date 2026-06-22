# Practitioner Guidelines: Uncertainty Quantification for MinHash Jaccard Estimates

## Summary of Findings

Based on comprehensive evaluation of 5 UQ methods across 3000 document pairs from 3 datasets:

### Coverage Probability (Target: 95%)
- **EVT-Gumbel**: 95.6% coverage - PASS
- **EVT-Weibull**: 94.8% coverage - PASS
- **Corrected Bootstrap**: 75.5% coverage - FAIL
- **Analytical Binomial**: 96.5% coverage - PASS
- **Bayesian**: 94.8% coverage - PASS

### Recommendations

1. **For Production Use**:
   - Use **Analytical Binomial** or **Bayesian** methods for well-calibrated 95% CIs
   - These achieve 96.5% and 94.8% coverage respectively (within 2% of target)

2. **Avoid EVT Methods** for short text (<100 shingles):
   - EVT-Gumbel: Coverage is acceptable but theoretically unjustified
   - EVT-Weibull: Similar issues - distributional assumptions violated

3. **Corrected Bootstrap** needs more resamples:
   - Only 75.5% coverage with 1000 resamples
   - Increase bootstrap samples to 5000+ for better calibration

4. **Method Selection by Use Case**:
   - **Speed critical**: Analytical Binomial (0.01ms) or Bayesian (0.02ms)
   - **Theoretical rigor**: Analytical Binomial (exact Clopper-Pearson)
   - **Avoid**: EVT methods for short text documents

### Technical Details
- **Evaluation**: 3000 document pairs (1000 per dataset)
- **Datasets**: Tweet sentiment, Tweet emoji, AG News headlines
- **MinHash**: 128 hash functions, k=3 character shingles
