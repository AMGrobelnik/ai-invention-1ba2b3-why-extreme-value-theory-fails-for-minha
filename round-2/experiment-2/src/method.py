#!/usr/bin/env python3
"""Alternative Uncertainty Quantification for MinHash: Binomial and Bayesian Methods.

Implements and evaluates two practical alternatives to EVT-MinHash for uncertainty
quantification: (1) analytical binomial confidence intervals (Clopper-Pearson exact
and Wilson score) based on matching hash counts, and (2) Bayesian approach with Beta
prior informed by document length. Compares both methods against bootstrap baseline
on short text datasets.
"""

from loguru import logger
from pathlib import Path
import json
import numpy as np
import hashlib
import time
import gc
from collections import defaultdict
from scipy import stats
from scipy.stats import beta, binom
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")


class MinHash:
    """MinHash implementation with k hash functions."""

    def __init__(self, k=128, seed=42):
        self.k = k
        self.seed = seed
        self.seeds = [seed + i for i in range(k)]

    def get_shingles(self, text, k=3):
        """Generate k-shingles from text."""
        words = text.lower().split()
        if len(words) < k:
            return set()
        shingles = set()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)
        return shingles

    def compute_signature(self, text):
        """Compute MinHash signature for a document."""
        shingles = self.get_shingles(text)
        if not shingles:
            return [float('inf')] * self.k

        signature = []
        for seed in self.seeds:
            min_hash = float('inf')
            for shingle in shingles:
                h = hashlib.md5(f"{seed}_{shingle}".encode()).hexdigest()
                h_int = int(h[:8], 16)
                h_normalized = h_int / (2**32)
                min_hash = min(min_hash, h_normalized)
            signature.append(min_hash)
        return signature

    def compute_signature_fast(self, text):
        """Faster MinHash using numpy for batch hash computation."""
        shingles = self.get_shingles(text)
        if not shingles:
            return np.full(self.k, float('inf'))

        shingle_list = list(shingles)
        signature = np.full(self.k, float('inf'))

        for i, seed in enumerate(self.seeds):
            min_val = float('inf')
            for shingle in shingle_list:
                h = hashlib.md5(f"{seed}_{shingle}".encode()).hexdigest()
                h_int = int(h[:8], 16)
                h_norm = h_int / (2**32)
                min_val = min(min_val, h_norm)
            signature[i] = min_val

        return signature


def true_jaccard(set_a, set_b):
    """Compute true Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def get_shingle_set(text, k=3):
    """Get shingle set for true Jaccard computation."""
    words = text.lower().split()
    if len(words) < k:
        return set()
    shingles = set()
    for i in range(len(words) - k + 1):
        shingles.add(' '.join(words[i:i+k]))
    return shingles


class BinomialCI:
    """Binomial confidence intervals for MinHash matching counts."""

    @staticmethod
    def clopper_pearson(x, n, confidence=0.95):
        """Clopper-Pearson exact confidence interval for binomial proportion."""
        alpha = 1 - confidence

        if x == 0:
            lower = 0.0
            upper = 1 - (alpha/2)**(1/n)
        elif x == n:
            lower = (alpha/2)**(1/n)
            upper = 1.0
        else:
            lower = beta.ppf(alpha/2, x, n - x + 1)
            upper = beta.ppf(1 - alpha/2, x + 1, n - x)

        return (lower, upper)

    @staticmethod
    def wilson_score(x, n, confidence=0.95):
        """Wilson score interval for binomial proportion."""
        p_hat = x / n
        z = stats.norm.ppf((1 + confidence) / 2)

        denominator = 1 + z**2/n
        center = (p_hat + z**2/(2*n)) / denominator
        margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator

        lower = max(0, center - margin)
        upper = min(1, center + margin)

        return (lower, upper)

    @staticmethod
    def jeffreys(x, n, confidence=0.95):
        """Jeffreys interval (Bayesian with Jeffreys prior Beta(0.5, 0.5))."""
        alpha = 1 - confidence
        posterior = beta(x + 0.5, n - x + 0.5)
        lower = posterior.ppf(alpha/2)
        upper = posterior.ppf(1 - alpha/2)
        return (lower, upper)


class BayesianMinHash:
    """Bayesian approach to MinHash uncertainty quantification."""

    @staticmethod
    def compute_prior_params(doc_length, method='uniform'):
        """Compute Beta prior parameters based on document characteristics."""
        if method == 'uniform':
            return (1.0, 1.0)
        elif method == 'jeffreys':
            return (0.5, 0.5)
        elif method == 'length_informed':
            shingle_count = doc_length
            if shingle_count < 50:
                return (0.7, 0.7)
            elif shingle_count < 100:
                return (1.0, 1.0)
            else:
                return (2.0, 2.0)
        else:
            return (1.0, 1.0)

    @staticmethod
    def compute_posterior(x, n, alpha_prior, beta_prior):
        """Compute posterior distribution given binomial likelihood."""
        alpha_post = alpha_prior + x
        beta_post = beta_prior + n - x
        return beta(alpha_post, beta_post)

    @staticmethod
    def credible_interval(x, n, alpha_prior, beta_prior, confidence=0.95):
        """Compute credible interval from posterior."""
        posterior = BayesianMinHash.compute_posterior(x, n, alpha_prior, beta_prior)
        alpha = 1 - confidence
        lower = posterior.ppf(alpha/2)
        upper = posterior.ppf(1 - alpha/2)
        return (lower, upper)

    @staticmethod
    def posterior_mean(x, n, alpha_prior, beta_prior):
        """Compute posterior mean (Bayes estimate of J)."""
        alpha_post = alpha_prior + x
        beta_post = beta_prior + n - x
        return alpha_post / (alpha_post + beta_post)


class BootstrapMinHash:
    """Bootstrap confidence intervals for MinHash."""

    @staticmethod
    def bootstrap_ci(sig_a, sig_b, k, n_bootstrap=1000, confidence=0.95):
        """Compute bootstrap CI for Jaccard estimate."""
        observed_matches = sum(1 for i in range(k) if abs(sig_a[i] - sig_b[i]) < 1e-10)
        observed_jaccard = observed_matches / k

        bootstrap_jaccards = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(k, k, replace=True)
            matches = sum(1 for i in indices if abs(sig_a[i] - sig_b[i]) < 1e-10)
            bootstrap_jaccards.append(matches / k)

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_jaccards, 100 * alpha/2)
        upper = np.percentile(bootstrap_jaccards, 100 * (1 - alpha/2))

        return (lower, upper)


def load_data(data_path):
    """Load and parse documents from JSON file."""
    logger.info(f"Loading data from {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)

    documents = []
    for dataset in data['datasets']:
        for example in dataset['examples']:
            doc = {
                'doc_id': example['metadata_doc_id'],
                'text': example['input'],
                'source': example['metadata_source'],
                'word_count': example['metadata_word_count'],
                'shingle_count': example['metadata_shingle_count']
            }
            documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents")
    return documents


def generate_document_pairs(documents, n_pairs=1000, min_jaccard=0.0, max_jaccard=1.0, seed=42):
    """Generate document pairs with known Jaccard similarity."""
    pairs = []
    np.random.seed(seed)

    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 10:
        i, j = np.random.choice(len(documents), 2, replace=False)
        doc_i = documents[i]
        doc_j = documents[j]

        shingles_i = get_shingle_set(doc_i['text'])
        shingles_j = get_shingle_set(doc_j['text'])
        jaccard = true_jaccard(shingles_i, shingles_j)

        if min_jaccard <= jaccard <= max_jaccard:
            pairs.append({
                'doc_i': doc_i,
                'doc_j': doc_j,
                'true_jaccard': jaccard,
                'index_i': i,
                'index_j': j
            })

        attempts += 1

    return pairs


def evaluate_coverage(pairs, k_values=[32, 64, 128], n_runs=100):
    """Evaluate coverage probability of CI methods."""
    results = {
        'binomial_clopper_pearson': [],
        'binomial_wilson': [],
        'bayesian_uniform': [],
        'bayesian_length_informed': [],
        'bootstrap': []
    }

    for pair_idx, pair in enumerate(pairs):
        doc_i = pair['doc_i']
        doc_j = pair['doc_j']
        true_J = pair['true_jaccard']

        for k in k_values:
            match_counts = []

            for run in range(n_runs):
                mh = MinHash(k=k, seed=run)
                sig_i = mh.compute_signature(doc_i['text'])
                sig_j = mh.compute_signature(doc_j['text'])

                matches = sum(1 for a, b in zip(sig_i, sig_j)
                              if abs(a - b) < 1e-10)
                match_counts.append(matches)

            for method_name in results.keys():
                coverages = []
                widths = []

                for x in match_counts:
                    if method_name == 'binomial_clopper_pearson':
                        lower, upper = BinomialCI.clopper_pearson(x, k)
                    elif method_name == 'binomial_wilson':
                        lower, upper = BinomialCI.wilson_score(x, k)
                    elif method_name == 'bayesian_uniform':
                        lower, upper = BayesianMinHash.credible_interval(
                            x, k, 1.0, 1.0)
                    elif method_name == 'bayesian_length_informed':
                        avg_shingles = (doc_i['shingle_count'] +
                                       doc_j['shingle_count']) / 2
                        alpha, beta = BayesianMinHash.compute_prior_params(
                            avg_shingles, 'length_informed')
                        lower, upper = BayesianMinHash.credible_interval(
                            x, k, alpha, beta)
                    elif method_name == 'bootstrap':
                        mh = MinHash(k=k, seed=0)
                        sig_i = mh.compute_signature(doc_i['text'])
                        sig_j = mh.compute_signature(doc_j['text'])
                        lower, upper = BootstrapMinHash.bootstrap_ci(
                            sig_i, sig_j, k, n_bootstrap=1000)

                    covered = (lower <= true_J <= upper)
                    coverages.append(covered)
                    widths.append(upper - lower)

                results[method_name].append({
                    'pair_idx': pair_idx,
                    'k': k,
                    'true_jaccard': true_J,
                    'coverage': np.mean(coverages),
                    'avg_width': np.mean(widths),
                    'match_counts': match_counts
                })

    return results


def benchmark_computation_time():
    """Benchmark computation time for each method."""
    k = 128
    n_runs = 100

    doc1 = "This is a test document with some words for benchmarking."
    doc2 = "This is another test document with different words."

    mh = MinHash(k=k, seed=0)
    sig1 = mh.compute_signature(doc1)
    sig2 = mh.compute_signature(doc2)

    match_count = sum(1 for a, b in zip(sig1, sig2)
                      if abs(a - b) < 1e-10)

    results = {}

    start = time.time()
    for _ in range(n_runs):
        BinomialCI.clopper_pearson(match_count, k)
    results['clopper_pearson'] = (time.time() - start) / n_runs

    start = time.time()
    for _ in range(n_runs):
        BinomialCI.wilson_score(match_count, k)
    results['wilson'] = (time.time() - start) / n_runs

    start = time.time()
    for _ in range(n_runs):
        BayesianMinHash.credible_interval(match_count, k, 1.0, 1.0)
    results['bayesian'] = (time.time() - start) / n_runs

    start = time.time()
    for _ in range(n_runs):
        BootstrapMinHash.bootstrap_ci(sig1, sig2, k, n_bootstrap=100)
    results['bootstrap_100'] = (time.time() - start) / n_runs

    start = time.time()
    for _ in range(n_runs):
        BootstrapMinHash.bootstrap_ci(sig1, sig2, k, n_bootstrap=1000)
    results['bootstrap_1000'] = (time.time() - start) / n_runs

    return results


def analyze_results(results, timing):
    """Analyze and summarize results."""
    analysis = {
        'coverage_summary': {},
        'width_summary': {},
        'timing_summary': timing
    }

    for method, result_list in results.items():
        coverage_by_k = defaultdict(list)
        width_by_k = defaultdict(list)

        for r in result_list:
            coverage_by_k[r['k']].append(r['coverage'])
            width_by_k[r['k']].append(r['avg_width'])

        analysis['coverage_summary'][method] = {
            k: {
                'mean': np.mean(v),
                'std': np.std(v),
                'target': 0.95
            }
            for k, v in coverage_by_k.items()
        }

        analysis['width_summary'][method] = {
            k: {
                'mean': np.mean(v),
                'std': np.std(v)
            }
            for k, v in width_by_k.items()
        }

    return analysis


@logger.catch(reraise=True)
def main():
    """Main experiment to evaluate all methods."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-pairs', type=int, default=500, help='Number of document pairs')
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs per pair')
    parser.add_argument('--k-values', type=int, nargs='+', default=[32, 64, 128], help='k values')
    parser.add_argument('--data-file', type=str, default='full_data_out.json', help='Data file path')
    args = parser.parse_args()

    logger.info(f"Starting Alternative Uncertainty Quantification Experiment with n_pairs={args.n_pairs}, n_runs={args.n_runs}")

    # Load data
    logger.info("Loading data...")
    data_path = Path(args.data_file)
    documents = load_data(data_path)

    # Generate document pairs
    logger.info("Generating document pairs...")
    pairs = generate_document_pairs(documents, n_pairs=args.n_pairs)
    logger.info(f"Generated {len(pairs)} pairs")

    # Evaluate coverage
    logger.info("Evaluating coverage...")
    results = evaluate_coverage(pairs, k_values=args.k_values, n_runs=args.n_runs)

    # Free memory
    del documents
    gc.collect()

    # Benchmark computation
    logger.info("Benchmarking computation time...")
    timing_results = benchmark_computation_time()

    # Analyze results
    logger.info("Analyzing results...")
    analysis = analyze_results(results, timing_results)

    # Save results in exp_gen_sol_out schema format
    output_datasets = []

    # Group results by source dataset
    for pair_idx, pair in enumerate(pairs):
        doc_i = pair['doc_i']
        doc_j = pair['doc_j']

        # Find results for this pair
        pair_results = {}
        for method_name in results.keys():
            method_results = [r for r in results[method_name] if r['pair_idx'] == pair_idx]
            if method_results:
                pair_results[method_name] = method_results[0]

        # Create example
        text_i = doc_i['text'][:100]
        text_j = doc_j['text'][:100]
        example = {
            'input': f"Doc i: {text_i}... | Doc j: {text_j}...",
            'output': str(pair['true_jaccard']),
            'metadata_doc_i_id': doc_i['doc_id'],
            'metadata_doc_j_id': doc_j['doc_id'],
            'metadata_source': doc_i['source'],
            'metadata_true_jaccard': pair['true_jaccard']
        }

        # Add predictions for each method
        for method_name, result in pair_results.items():
            example[f'predict_{method_name}'] = json.dumps({
                'coverage': result['coverage'],
                'avg_width': result['avg_width'],
                'k': result['k']
            })

        # Add to dataset group
        dataset_name = doc_i['source']
        found = False
        for ds in output_datasets:
            if ds['dataset'] == dataset_name:
                ds['examples'].append(example)
                found = True
                break

        if not found:
            output_datasets.append({
                'dataset': dataset_name,
                'examples': [example]
            })

    output = {
        'metadata': {
            'experiment': 'alternative_uncertainty_quantification',
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'parameters': {
                'k_values': args.k_values,
                'n_pairs': len(pairs),
                'n_runs': args.n_runs
            }
        },
        'datasets': output_datasets
    }

    output_path = Path("method_out.json")
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Experiment complete. Results saved to {output_path}")

    return output


if __name__ == '__main__':
    main()
