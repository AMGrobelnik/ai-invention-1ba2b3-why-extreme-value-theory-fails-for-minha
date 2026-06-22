#!/usr/bin/env python3
"""EVT-MinHash Distribution Verification and Bootstrap Baseline.

Empirically verify whether MinHash signature values follow a Gumbel distribution
for short text documents (10-100 shingles), implement bootstrap baseline for
Jaccard similarity confidence intervals, and compare computational costs.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import os
import gc
import time
import random
import hashlib
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict

# Lazy imports - will be loaded when needed
_scipy_stats = None
_plt = None
_sns = None
_datasets_available = False

def get_scipy_stats():
    """Lazy load scipy.stats."""
    global _scipy_stats
    if _scipy_stats is None:
        from scipy.stats import gumbel_l, kstest, anderson
        _scipy_stats = {
            'gumbel_l': gumbel_l,
            'kstest': kstest,
            'anderson': anderson
        }
    return _scipy_stats

def get_plt():
    """Lazy load matplotlib."""
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt

def get_sns():
    """Lazy load seaborn."""
    global _sns
    if _sns is None:
        import seaborn as sns
        _sns = sns
    return _sns

def get_datasets():
    """Lazy load datasets library."""
    global _datasets_available
    if not _datasets_available:
        try:
            from datasets import load_dataset
            _datasets_available = True
            return load_dataset
        except ImportError:
            logger.warning("datasets not installed, will use synthetic data")
            return None
    # Return the loaded function
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        return None


# ============================================================================
# Configuration and Constants
# ============================================================================

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    num_hashes: int = 128
    k_shingle: int = 3
    shingle_type: str = 'char'  # 'char' or 'word'
    num_bootstrap: int = 1000
    confidence_level: float = 0.95
    num_document_pairs: int = 1000
    shingle_count_ranges: List[Tuple[int, int]] = None
    datasets: List[str] = None
    random_seed: int = 42

    def __post_init__(self):
        if self.shingle_count_ranges is None:
            self.shingle_count_ranges = [(10, 30), (30, 50), (50, 70), (70, 100)]
        if self.datasets is None:
            self.datasets = ['tweets', 'sms', 'headlines']


# ============================================================================
# Hardware Detection
# ============================================================================

def detect_hardware():
    """Detect hardware resources in container-aware manner."""
    import math

    def _detect_cpus() -> int:
        """Detect actual CPU allocation."""
        try:
            parts = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip()
            q = int(parts)
            p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
            if q > 0:
                return math.ceil(q / p)
        except (FileNotFoundError, ValueError):
            pass
        try:
            return len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            pass
        return os.cpu_count() or 1

    def _container_ram_gb() -> float:
        """Read RAM limit from cgroup."""
        try:
            v = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip()
            if int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
        return 28.0  # Default from hardware detection

    cpus = _detect_cpus()
    ram_gb = _container_ram_gb()

    logger.info(f"Detected hardware: {cpus} CPUs, {ram_gb:.1f} GB RAM")
    return cpus, ram_gb


# ============================================================================
# MinHash Implementation
# ============================================================================

class MinHash:
    """MinHash implementation for Jaccard similarity estimation."""

    def __init__(self, num_hashes: int = 128, seed: int = 42):
        """Initialize MinHash with specified number of hash functions."""
        self.num_hashes = num_hashes
        self.seed = seed
        self.hash_functions = self._generate_hash_functions()

    def _generate_hash_functions(self):
        """Generate independent hash functions using different seeds."""
        hash_funcs = []
        for i in range(self.num_hashes):
            seed_i = self.seed + i * 1000
            hash_funcs.append(lambda x, s=seed_i: self._hash(x, s))
        return hash_funcs

    def _hash(self, shingle: str, seed: int) -> int:
        """Hash a shingle with given seed."""
        combined = f"{seed}_{shingle}".encode('utf-8')
        return int(hashlib.md5(combined).hexdigest(), 16)

    def _get_shingles(self, text: str, k: int = 3, shingle_type: str = 'char') -> set:
        """Extract shingles from text."""
        if shingle_type == 'char':
            text = text.lower()
            if len(text) < k:
                return {text}
            return set(text[i:i+k] for i in range(len(text) - k + 1))
        elif shingle_type == 'word':
            tokens = text.lower().split()
            if len(tokens) < k:
                return {' '.join(tokens)}
            return set(' '.join(tokens[i:i+k]) for i in range(len(tokens) - k + 1))
        else:
            raise ValueError(f"Unknown shingle_type: {shingle_type}")

    def compute_signature(self, text: str, k: int = 3, shingle_type: str = 'char') -> Tuple[List[int], int]:
        """Compute MinHash signature for text.

        Returns:
            Tuple of (signature, shingle_count)
        """
        shingles = self._get_shingles(text, k, shingle_type)
        if not shingles:
            return [0] * self.num_hashes, 0

        signature = []
        for i in range(self.num_hashes):
            min_hash = min(self.hash_functions[i](shingle) for shingle in shingles)
            signature.append(min_hash)

        return signature, len(shingles)

    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have same length")
        matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return matches / len(sig1)

    def exact_jaccard(self, text1: str, text2: str, k: int = 3, shingle_type: str = 'char') -> float:
        """Compute exact Jaccard similarity."""
        shingles1 = self._get_shingles(text1, k, shingle_type)
        shingles2 = self._get_shingles(text2, k, shingle_type)
        if not shingles1 and not shingles2:
            return 1.0
        union = shingles1 | shingles2
        if not union:
            return 1.0
        return len(shingles1 & shingles2) / len(union)


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

class BootstrapCI:
    """Bootstrap confidence intervals for Jaccard similarity."""

    def __init__(self, num_bootstrap: int = 1000, random_seed: int = 42):
        """Initialize bootstrap CI calculator."""
        self.num_bootstrap = num_bootstrap
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

    def compute_bootstrap_ci(
        self,
        doc1: str,
        doc2: str,
        minhash: MinHash,
        k: int = 3,
        shingle_type: str = 'char',
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval for Jaccard similarity.

        Returns:
            Tuple of (lower_ci, upper_ci, mean_bootstrap)
        """
        shingles1 = list(minhash._get_shingles(doc1, k, shingle_type))
        shingles2 = list(minhash._get_shingles(doc2, k, shingle_type))

        if not shingles1 or not shingles2:
            return 0.0, 0.0, 0.0

        bootstrap_similarities = []

        for b in range(self.num_bootstrap):
            # Resample shingles with replacement
            sample1 = [self.rng.choice(shingles1) for _ in range(len(shingles1))]
            sample2 = [self.rng.choice(shingles2) for _ in range(len(shingles2))]

            # Compute Jaccard from resampled sets
            set1, set2 = set(sample1), set(sample2)
            union = set1 | set2
            if union:
                jaccard = len(set1 & set2) / len(union)
            else:
                jaccard = 1.0
            bootstrap_similarities.append(jaccard)

        # Compute percentile CI
        alpha = 1 - confidence
        lower = float(np.percentile(bootstrap_similarities, (alpha/2)*100))
        upper = float(np.percentile(bootstrap_similarities, (1-alpha/2)*100))
        mean_boot = float(np.mean(bootstrap_similarities))

        return lower, upper, mean_boot


# ============================================================================
# Gumbel Distribution Fitting
# ============================================================================

class GumbelFitter:
    """Fit EVT distributions to MinHash signature values."""

    def __init__(self):
        self.fit_results = {}

    def fit_distribution(self, data: np.ndarray, dist_name: str = 'gumbel') -> Dict[str, Any]:
        """Fit distribution using MLE.

        Args:
            data: Data to fit
            dist_name: 'gumbel' or 'weibull' (for minima of bounded distributions)

        Returns:
            Dictionary with fit parameters and test statistics
        """
        if len(data) < 10:
            logger.warning(f"Insufficient data for distribution fit: {len(data)} samples")
            return None

        try:
            stats = get_scipy_stats()

            # Ensure data is float64 for scipy
            data = np.array(data, dtype=np.float64)

            # Check for valid data
            if not np.all(np.isfinite(data)):
                logger.warning("Data contains non-finite values, skipping fit")
                return None

            if dist_name == 'gumbel':
                return self._fit_gumbel(data, stats)
            elif dist_name == 'weibull':
                return self._fit_weibull(data, stats)
            else:
                logger.error(f"Unknown distribution: {dist_name}")
                return None

        except Exception as e:
            logger.error(f"Distribution fitting failed: {e}")
            return None

    def _fit_gumbel(self, data: np.ndarray, stats: dict) -> Dict[str, Any]:
        """Fit Gumbel distribution (Type I EVT)."""
        gumbel_l = stats['gumbel_l']
        kstest = stats['kstest']
        anderson = stats['anderson']

        # Fit Gumbel (for minima)
        params = gumbel_l.fit(data)
        loc, scale = params

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = kstest(data, lambda x: gumbel_l.cdf(x, loc=loc, scale=scale))

        # Anderson-Darling test
        ad_result = anderson(data, dist='gumbel')

        return {
            'distribution': 'gumbel',
            'loc': float(loc),
            'scale': float(scale),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'ad_statistic': float(ad_result.statistic),
            'ad_critical_values': ad_result.critical_values.tolist(),
            'ad_significance_level': ad_result.significance_level.tolist(),
            'num_samples': len(data)
        }

    def _fit_weibull(self, data: np.ndarray, stats: dict) -> Dict[str, Any]:
        """Fit Weibull distribution (Type III EVT for bounded distributions)."""
        from scipy.stats import weibull_min
        kstest = stats['kstest']

        # Fit Weibull minimum (for minima of bounded distributions)
        # weibull_min is the Weibull distribution for minima
        params = weibull_min.fit(data, floc=0)  # Fix location at 0 for bounded [0,1]
        shape, loc, scale = params

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = kstest(data, lambda x: weibull_min.cdf(x, shape, loc=loc, scale=scale))

        return {
            'distribution': 'weibull',
            'shape': float(shape),
            'loc': float(loc),
            'scale': float(scale),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'num_samples': len(data)
        }

    def fit_gumbel(self, data: np.ndarray) -> Dict[str, Any]:
        """Backward compatibility wrapper."""
        return self.fit_distribution(data, 'gumbel')

    def extract_minhash_values(
        self,
        docs: List[str],
        minhash: MinHash,
        k: int = 3,
        shingle_type: str = 'char'
    ) -> Dict[str, np.ndarray]:
        """Extract various MinHash values for distribution fitting."""
        all_min_values = []
        all_max_values = []
        all_mean_values = []

        for doc in docs:
            sig, shingle_count = minhash.compute_signature(doc, k, shingle_type)
            if shingle_count > 0:
                # Normalize hash values to [0, 1] by dividing by max MD5 value
                # MD5 produces 128-bit hash, max value is 2^128 - 1
                max_hash = 2**128 - 1
                sig_normalized = np.array(sig, dtype=np.float64) / max_hash
                
                all_min_values.append(np.min(sig_normalized))
                all_max_values.append(np.max(sig_normalized))
                all_mean_values.append(np.mean(sig_normalized))

        return {
            'min_values': np.array(all_min_values, dtype=np.float64),
            'max_values': np.array(all_max_values, dtype=np.float64),
            'mean_values': np.array(all_mean_values, dtype=np.float64)
        }


# ============================================================================
# Dataset Loading
# ============================================================================

class DatasetLoader:
    """Load and preprocess short-text datasets."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)
        np.random.seed(config.random_seed)

    def load_tweets(self, max_docs: int = 2000) -> List[str]:
        """Load tweet dataset."""
        docs = []
        load_dataset = get_datasets()

        try:
            if load_dataset is not None:
                # Try to load from HuggingFace
                dataset = load_dataset("sentiment140", split="train", streaming=True)
                count = 0
                for example in dataset:
                    if 'text' in example and example['text']:
                        text = self._preprocess_text(example['text'])
                        word_count = len(text.split())
                        if 10 <= word_count <= 100:
                            docs.append(text)
                            count += 1
                            if count >= max_docs:
                                break
        except Exception as e:
            logger.warning(f"Failed to load tweets dataset: {e}")

        if len(docs) < 100:
            logger.info("Generating synthetic tweets")
            docs = self._generate_synthetic_docs(max_docs, 'tweet')

        logger.info(f"Loaded {len(docs)} tweets")
        return docs

    def load_sms(self, max_docs: int = 2000) -> List[str]:
        """Load SMS dataset."""
        docs = []
        load_dataset = get_datasets()

        try:
            if load_dataset is not None:
                # Try multiple possible SMS dataset names
                for dataset_name in ["sms_spam_classification", "uci/sms_spam_collection", "sms_spam"]:
                    try:
                        dataset = load_dataset(dataset_name, split="train")
                        for example in dataset:
                            if 'text' in example and example['text']:
                                text = self._preprocess_text(example['text'])
                                word_count = len(text.split())
                                if 10 <= word_count <= 100:
                                    docs.append(text)
                        if len(docs) > 0:
                            break
                    except Exception:
                        continue

                if len(docs) > max_docs:
                    docs = docs[:max_docs]
        except Exception as e:
            logger.warning(f"Failed to load SMS dataset: {e}")

        if len(docs) < 100:
            logger.info("Generating synthetic SMS messages")
            docs = self._generate_synthetic_docs(max_docs, 'sms')

        logger.info(f"Loaded {len(docs)} SMS messages")
        return docs

    def load_headlines(self, max_docs: int = 2000) -> List[str]:
        """Load news headlines dataset."""
        docs = []
        load_dataset = get_datasets()

        try:
            if load_dataset is not None:
                # Try to load AG News or similar
                dataset = load_dataset("ag_news", split="train")
                for example in dataset:
                    if 'text' in example and example['text']:
                        text = self._preprocess_text(example['text'])
                        word_count = len(text.split())
                        if 10 <= word_count <= 100:
                            docs.append(text)
                if len(docs) > max_docs:
                    docs = docs[:max_docs]
        except Exception as e:
            logger.warning(f"Failed to load headlines dataset: {e}")

        if len(docs) < 100:
            logger.info("Generating synthetic headlines")
            docs = self._generate_synthetic_docs(max_docs, 'headline')

        logger.info(f"Loaded {len(docs)} headlines")
        return docs

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text: lowercase, remove URLs and special chars."""
        import re
        text = text.lower()
        text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def _generate_synthetic_docs(self, num_docs: int, doc_type: str) -> List[str]:
        """Generate synthetic documents for testing."""
        docs = []
        templates = {
            'tweet': [
                "just had the best {food} ever at {place}",
                "can't believe {sports_team} won the {event} today",
                "feeling so {emotion} right now about {topic}",
                "my {device} is acting up again need to fix it",
                "great day for {activity} with {person}",
            ],
            'sms': [
                "hey are you coming to {place} today",
                "don't forget to bring {item} for the {event}",
                "can you pick up {item} on your way home",
                "meeting moved to {time} please confirm",
                "happy birthday {person} have a great day",
            ],
            'headline': [
                "{company} announces new {product} for {year}",
                "{person} wins {award} for {achievement}",
                "study finds {finding} in {field} research",
                "{event} expected to impact {industry} sector",
                "new {technology} could revolutionize {field}",
            ]
        }

        words = {
            'food': ['pizza', 'sushi', 'burger', 'pasta', 'salad'],
            'place': ['restaurant', 'cafe', 'home', 'park', 'mall'],
            'sports_team': ['Lakers', 'Yankees', 'Cowboys', 'Warriors'],
            'event': ['game', 'match', 'championship', 'tournament'],
            'emotion': ['happy', 'sad', 'excited', 'nervous', 'grateful'],
            'topic': ['work', 'school', 'life', 'family', 'friends'],
            'device': ['phone', 'laptop', 'tablet', 'watch', 'computer'],
            'activity': ['hiking', 'swimming', 'reading', 'coding', 'running'],
            'person': ['mom', 'dad', 'friend', 'sister', 'brother'],
            'item': ['groceries', 'books', 'gifts', 'supplies', 'snacks'],
            'time': ['3pm', 'noon', '5pm', 'morning', 'evening'],
            'company': ['Apple', 'Google', 'Microsoft', 'Amazon', 'Tesla'],
            'product': ['phone', 'laptop', 'software', 'service', 'device'],
            'year': ['2024', '2025', '2026'],
            'award': ['prize', 'medal', 'trophy', 'recognition', 'grant'],
            'achievement': ['research', 'innovation', 'discovery', 'breakthrough'],
            'finding': ['results', 'patterns', 'trends', 'correlations'],
            'field': ['science', 'technology', 'medicine', 'engineering'],
            'industry': ['tech', 'healthcare', 'finance', 'education'],
            'technology': ['AI', 'blockchain', 'quantum', 'renewable'],
        }

        import re
        template_list = templates.get(doc_type, templates['tweet'])

        for i in range(num_docs):
            template = self.rng.choice(template_list)
            # Find all placeholders
            placeholders = re.findall(r'\{(\w+)\}', template)
            # Replace with random choices
            text = template
            for ph in placeholders:
                if ph in words:
                    text = text.replace(f'{{{ph}}}', self.rng.choice(words[ph]), 1)
            docs.append(text)

        return docs

    def filter_by_shingle_count(
        self,
        docs: List[str],
        min_shingles: int,
        max_shingles: int,
        k: int = 3,
        shingle_type: str = 'char'
    ) -> List[str]:
        """Filter documents by shingle count."""
        minhash = MinHash(num_hashes=1)  # Just need shingle computation
        filtered = []
        for doc in docs:
            shingles = minhash._get_shingles(doc, k, shingle_type)
            count = len(shingles)
            if min_shingles <= count <= max_shingles:
                filtered.append(doc)
        return filtered

    def generate_document_pairs(
        self,
        docs: List[str],
        n_pairs: int,
        similarity_range: Tuple[float, float] = (0.0, 1.0)
    ) -> List[Tuple[str, str]]:
        """Generate random document pairs."""
        if len(docs) < 2:
            return []

        pairs = []
        for _ in range(n_pairs):
            idx1, idx2 = self.rng.sample(range(len(docs)), 2)
            pairs.append((docs[idx1], docs[idx2]))

        return pairs


# ============================================================================
# Visualization
# ============================================================================

class Visualizer:
    """Generate visualization plots."""

    @staticmethod
    def plot_qq_distribution(data: np.ndarray, fit_result: Dict, output_path: str, title: str = "QQ Plot"):
        """Create QQ plot comparing data to fitted distribution."""
        plt = get_plt()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        dist_name = fit_result.get('distribution', 'unknown')

        if dist_name == 'gumbel':
            from scipy.stats import gumbel_l
            params = (fit_result['loc'], fit_result['scale'])
            dist_obj = gumbel_l(loc=params[0], scale=params[1])

            # QQ Plot
            from scipy import stats as scipy_stats
            scipy_stats.probplot(data, dist='gumbel_l', sparams=params, plot=axes[0])
            axes[0].set_title(f"{title} - Gumbel QQ Plot")

            # Histogram with fitted PDF
            axes[1].hist(data, bins=50, density=True, alpha=0.7, label='Empirical')
            x = np.linspace(min(data), max(data), 1000)
            pdf = gumbel_l.pdf(x, loc=params[0], scale=params[1])
            axes[1].plot(x, pdf, 'r-', lw=2, label='Gumbel PDF')
            axes[1].set_title(f"{title} - Histogram with Gumbel PDF")
            axes[1].legend()

        elif dist_name == 'weibull':
            from scipy.stats import weibull_min
            shape = fit_result['shape']
            loc = fit_result['loc']
            scale = fit_result['scale']
            dist_obj = weibull_min(shape, loc=loc, scale=scale)

            # QQ Plot using theoretical quantiles
            from scipy import stats as scipy_stats
            # Calculate theoretical quantiles
            n = len(data)
            p = np.arange(1, n + 1) / (n + 1)
            theoretical_quantiles = dist_obj.ppf(p)
            axes[0].scatter(np.sort(theoretical_quantiles), np.sort(data), alpha=0.5)
            # Add diagonal line
            lims = [min(min(data), min(theoretical_quantiles)), max(max(data), max(theoretical_quantiles))]
            axes[0].plot(lims, lims, 'r--', alpha=0.75, zorder=0)
            axes[0].set_xlabel('Theoretical Quantiles')
            axes[0].set_ylabel('Ordered Values')
            axes[0].set_title(f"{title} - Weibull QQ Plot")

            # Histogram with fitted PDF
            axes[1].hist(data, bins=50, density=True, alpha=0.7, label='Empirical')
            x = np.linspace(min(data), max(data), 1000)
            pdf = weibull_min.pdf(x, shape, loc=loc, scale=scale)
            axes[1].plot(x, pdf, 'r-', lw=2, label='Weibull PDF')
            axes[1].set_title(f"{title} - Histogram with Weibull PDF")
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved distribution QQ plot to {output_path}")

    @staticmethod
    def plot_cost_comparison(cost_data: Dict, output_path: str):
        """Plot computational cost comparison."""
        plt = get_plt()

        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(cost_data.keys())
        times = list(cost_data.values())

        ax.bar(methods, times, color=['blue', 'orange', 'green'])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Computational Cost Comparison')
        ax.set_xticklabels(methods, rotation=45, ha='right')

        # Add value labels on bars
        for i, v in enumerate(times):
            ax.text(i, v + 0.01, f'{v:.3f}s', ha='center')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved cost comparison plot to {output_path}")


# ============================================================================
# Main Experiment
# =========================================================================

@dataclass
class ExperimentResults:
    """Container for experiment results."""
    experiment_config: Dict
    gumbel_fit_results: List[Dict]
    bootstrap_results: List[Dict]
    computational_cost: Dict
    visualizations: List[str]


class MinHashExperiment:
    """Main experiment class."""

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.minhash = MinHash(num_hashes=self.config.num_hashes, seed=self.config.random_seed)
        self.bootstrap = BootstrapCI(num_bootstrap=self.config.num_bootstrap, random_seed=self.config.random_seed)
        self.gumbel_fitter = GumbelFitter()
        self.dataset_loader = DatasetLoader(self.config)
        self.visualizer = Visualizer()
        self.results = ExperimentResults(
            experiment_config=asdict(self.config),
            gumbel_fit_results=[],
            bootstrap_results=[],
            computational_cost={},
            visualizations=[]
        )

    def run_experiment(self):
        """Run the full experiment pipeline."""
        logger.info("Starting MinHash EVT experiment")
        logger.info(f"Configuration: {asdict(self.config)}")

        # Load datasets
        datasets = {}
        for dataset_name in self.config.datasets:
            logger.info(f"Loading {dataset_name} dataset")
            if dataset_name == 'tweets':
                docs = self.dataset_loader.load_tweets()
            elif dataset_name == 'sms':
                docs = self.dataset_loader.load_sms()
            elif dataset_name == 'headlines':
                docs = self.dataset_loader.load_headlines()
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue

            if docs:
                datasets[dataset_name] = docs
                logger.info(f"Loaded {len(docs)} documents for {dataset_name}")

        if not datasets:
            logger.error("No datasets loaded, cannot proceed")
            return self.results

        # Run experiments for each dataset and shingle count range
        for dataset_name, docs in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")

            for shingle_range in self.config.shingle_count_ranges:
                min_s, max_s = shingle_range
                logger.info(f"  Shingle count range: {min_s}-{max_s}")

                # Filter documents by shingle count
                filtered_docs = self.dataset_loader.filter_by_shingle_count(
                    docs, min_s, max_s,
                    k=self.config.k_shingle,
                    shingle_type=self.config.shingle_type
                )

                if len(filtered_docs) < 10:
                    logger.warning(f"    Insufficient documents ({len(filtered_docs)}), skipping")
                    continue

                logger.info(f"    Filtered to {len(filtered_docs)} documents")

                # Generate document pairs
                pairs = self.dataset_loader.generate_document_pairs(
                    filtered_docs,
                    min(self.config.num_document_pairs, len(filtered_docs) * 2)
                )

                # Compute MinHash signatures and extract values for Gumbel fitting
                logger.info(f"    Computing MinHash signatures for {len(pairs)} pairs")
                minhash_values = self.gumbel_fitter.extract_minhash_values(
                    [p[0] for p in pairs] + [p[1] for p in pairs],
                    self.minhash,
                    k=self.config.k_shingle,
                    shingle_type=self.config.shingle_type
                )

                # Fit distributions to minimum signature values
                logger.info(f"    Fitting distributions (Gumbel and Weibull)")

                # Try Gumbel fit
                gumbel_result = self.gumbel_fitter.fit_distribution(
                    minhash_values['min_values'], 'gumbel'
                )

                # Try Weibull fit (often better for bounded distributions like normalized hashes)
                weibull_result = self.gumbel_fitter.fit_distribution(
                    minhash_values['min_values'], 'weibull'
                )

                # Select best fit based on KS p-value
                best_result = None
                if gumbel_result and weibull_result:
                    if gumbel_result['ks_pvalue'] > weibull_result['ks_pvalue']:
                        best_result = gumbel_result
                        best_result['selected'] = 'gumbel'
                    else:
                        best_result = weibull_result
                        best_result['selected'] = 'weibull'
                elif gumbel_result:
                    best_result = gumbel_result
                    best_result['selected'] = 'gumbel'
                elif weibull_result:
                    best_result = weibull_result
                    best_result['selected'] = 'weibull'

                if best_result:
                    best_result['dataset'] = dataset_name
                    best_result['shingle_range'] = shingle_range
                    best_result['gumbel_ks_pvalue'] = gumbel_result['ks_pvalue'] if gumbel_result else None
                    best_result['weibull_ks_pvalue'] = weibull_result['ks_pvalue'] if weibull_result else None
                    self.results.gumbel_fit_results.append(best_result)

                    # Generate QQ plot for best fit
                    if len(minhash_values['min_values']) > 0:
                        fig_path = f"figures/distribution_qq_{dataset_name}_{min_s}_{max_s}.png"
                        fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fig_path)
                        self.visualizer.plot_qq_distribution(
                            minhash_values['min_values'],
                            best_result,
                            fig_path,
                            title=f"Distribution Fit - {dataset_name} ({min_s}-{max_s} shingles)"
                        )
                        self.results.visualizations.append(fig_path)

                # Compute bootstrap CIs for subset
                logger.info(f"    Computing bootstrap CIs")
                bootstrap_subset = pairs[:min(100, len(pairs))]
                for i, (doc1, doc2) in enumerate(bootstrap_subset):
                    try:
                        lower, upper, mean = self.bootstrap.compute_bootstrap_ci(
                            doc1, doc2, self.minhash,
                            k=self.config.k_shingle,
                            shingle_type=self.config.shingle_type,
                            confidence=self.config.confidence_level
                        )
                        exact_jacc = self.minhash.exact_jaccard(
                            doc1, doc2,
                            k=self.config.k_shingle,
                            shingle_type=self.config.shingle_type
                        )
                        minhash_jacc = self.minhash.jaccard_similarity(
                            *[self.minhash.compute_signature(d, self.config.k_shingle, self.config.shingle_type)[0] for d in [doc1, doc2]]
                        )

                        self.results.bootstrap_results.append({
                            'dataset': dataset_name,
                            'shingle_range': shingle_range,
                            'pair_id': i,
                            'exact_jaccard': exact_jacc,
                            'minhash_jaccard': minhash_jacc,
                            'bootstrap_lower': lower,
                            'bootstrap_upper': upper,
                            'bootstrap_mean': mean,
                            'ci_contains_true': lower <= exact_jacc <= upper
                        })
                    except Exception as e:
                        logger.error(f"    Bootstrap failed for pair {i}: {e}")

                # Clear memory
                del filtered_docs, pairs, minhash_values
                gc.collect()

        # Computational cost comparison
        logger.info("Running computational cost comparison")
        self._run_cost_comparison(datasets)

        logger.info("Experiment complete")
        return self.results

    def _run_cost_comparison(self, datasets: Dict[str, List[str]]):
        """Compare computational costs of different methods."""
        cost_results = {}

        # Get a sample of documents
        sample_docs = []
        for docs in datasets.values():
            sample_docs.extend(docs[:50])
        if len(sample_docs) > 100:
            sample_docs = sample_docs[:100]

        if not sample_docs:
            return

        # Time MinHash signature computation
        logger.info("  Timing MinHash signature computation")
        start = time.time()
        for doc in sample_docs:
            self.minhash.compute_signature(doc, self.config.k_shingle, self.config.shingle_type)
        minhash_time = time.time() - start
        cost_results['minhash_signature'] = minhash_time / len(sample_docs)

        # Time Bootstrap CI computation (small bootstrap for timing)
        logger.info("  Timing Bootstrap CI computation")
        bootstrap_small = BootstrapCI(num_bootstrap=100, random_seed=self.config.random_seed)
        doc1, doc2 = sample_docs[0], sample_docs[1]
        start = time.time()
        for _ in range(10):
            bootstrap_small.compute_bootstrap_ci(
                doc1, doc2, self.minhash,
                k=self.config.k_shingle,
                shingle_type=self.config.shingle_type
            )
        bootstrap_time = time.time() - start
        cost_results['bootstrap_ci_100'] = bootstrap_time / 10

        # Extrapolate for full bootstrap
        cost_results['bootstrap_ci_1000'] = bootstrap_time / 10 * (1000 / 100)

        self.results.computational_cost = cost_results
        logger.info(f"  Cost results: {cost_results}")


# ============================================================================
# Main Entry Point
# ============================================================================

@logger.catch(reraise=True)
def main():
    """Main entry point."""
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

    # Detect hardware
    cpus, ram_gb = detect_hardware()

    # Set memory limits
    import resource
    ram_budget = int(min(ram_gb * 0.9 * 1e9, 25e9))  # 90% of available, max 25GB
    resource.setrlimit(resource.RLIMIT_AS, (ram_budget, ram_budget))
    logger.info(f"Set RAM limit to {ram_budget / 1e9:.1f} GB")

    # Create output directories
    Path("logs").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    # Run experiment
    config = ExperimentConfig(
        num_hashes=128,
        k_shingle=3,
        shingle_type='char',
        num_bootstrap=1000,
        num_document_pairs=1000,
        datasets=['tweets', 'sms', 'headlines']
    )

    experiment = MinHashExperiment(config)
    results = experiment.run_experiment()

    # Save results
    output_file = Path("method_out.json")
    output_data = {
        'experiment_config': results.experiment_config,
        'gumbel_fit_results': results.gumbel_fit_results,
        'bootstrap_results': results.bootstrap_results,
        'computational_cost': results.computational_cost,
        'visualizations': [os.path.basename(v) for v in results.visualizations]
    }

    output_file.write_text(json.dumps(output_data, indent=2))
    logger.info(f"Saved results to {output_file}")

    # Validate output against schema
    logger.info("Validating output JSON schema")
    try:
        skill_dir = Path("/ai-inventor/.claude/skills/aii-json")
        py = skill_dir / "../.ability_client_venv/bin/python"
        schema_script = skill_dir / "scripts/aii_json_validate_schema.py"

        if py.exists() and schema_script.exists():
            import subprocess
            result = subprocess.run(
                [str(py), str(schema_script), "--format", "exp_gen_sol_out", "--file", str(output_file.absolute())],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("Schema validation PASSED")
            else:
                logger.warning(f"Schema validation issues: {result.stdout}\n{result.stderr}")
    except Exception as e:
        logger.warning(f"Could not validate schema: {e}")

    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
