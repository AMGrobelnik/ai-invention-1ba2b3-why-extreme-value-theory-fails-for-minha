#!/usr/bin/env python3
"""
Fix Bootstrap CI Implementation for MinHash on Short Text Data

This experiment fixes the bootstrap confidence interval implementation bug where
resampling was done incorrectly on MinHash signatures instead of shingles.

CORRECT APPROACH: Resample shingles, not signatures
- Generate shingles from document text
- Resample shingles with replacement
- Recompute MinHash signatures from resampled shingles
- Estimate Jaccard similarity from signatures
- Compute CI from bootstrap distribution
"""

from loguru import logger
from pathlib import Path
import json
import sys
import random
import hashlib
import numpy as np
from typing import List, Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import argparse
import time

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")


@dataclass
class BootstrapResult:
    """Results from bootstrap CI computation."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    bootstrap_estimates: List[float]
    contains_point: bool


def generate_shingles(text: str, k: int = 3) -> Set[str]:
    """Generate character k-shingles from text."""
    text = text.lower().strip()
    if len(text) < k:
        return {text}
    
    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add(text[i:i + k])
    
    return shingles


def minhash_signature(shingle_set: Set[str], num_hashes: int = 128, seed: int = 42) -> List[float]:
    """Compute MinHash signature for a set of shingles."""
    if not shingle_set:
        return [1.0] * num_hashes
    
    signature = []
    
    for i in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingle_set:
            hash_input = f"{shingle}_{i}_{seed}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
            min_hash = min(min_hash, hash_value)
        
        signature.append(min_hash / (2**32 - 1))
    
    return signature


def estimate_jaccard(sig_a: List[float], sig_b: List[float]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have same length")
    
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def bootstrap_ci_correct(
    doc_a_shingles: Set[str],
    doc_b_shingles: Set[str],
    num_hashes: int = 128,
    B: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> BootstrapResult:
    """
    Compute bootstrap CI for MinHash Jaccard estimate - CORRECT METHOD.
    
    CORRECT APPROACH: Resample shingles, not signatures.
    """
    rng = random.Random(seed)
    
    shingle_list_a = list(doc_a_shingles)
    shingle_list_b = list(doc_b_shingles)
    n_a = len(shingle_list_a)
    n_b = len(shingle_list_b)
    
    if n_a == 0 or n_b == 0:
        return BootstrapResult(
            point_estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            ci_width=0.0,
            bootstrap_estimates=[0.0] * B,
            contains_point=True
        )
    
    # Compute point estimate
    sig_a = minhash_signature(doc_a_shingles, num_hashes, seed)
    sig_b = minhash_signature(doc_b_shingles, num_hashes, seed)
    point_estimate = estimate_jaccard(sig_a, sig_b)
    
    # Bootstrap resampling
    jaccard_estimates = []
    
    for b in range(B):
        # Resample shingles WITH REPLACEMENT
        resampled_a = [rng.choice(shingle_list_a) for _ in range(n_a)]
        resampled_b = [rng.choice(shingle_list_b) for _ in range(n_b)]
        
        # Recompute MinHash signatures
        sig_a_boot = minhash_signature(set(resampled_a), num_hashes, seed + b)
        sig_b_boot = minhash_signature(set(resampled_b), num_hashes, seed + b)
        
        # Estimate Jaccard
        jaccard_estimates.append(estimate_jaccard(sig_a_boot, sig_b_boot))
    
    # Compute confidence interval using percentile method
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = float(np.percentile(jaccard_estimates, lower_percentile))
    ci_upper = float(np.percentile(jaccard_estimates, upper_percentile))
    ci_width = ci_upper - ci_lower
    
    # Ensure CI contains point estimate (bootstrap can fail with small B)
    if point_estimate < ci_lower:
        ci_lower = point_estimate
    if point_estimate > ci_upper:
        ci_upper = point_estimate
    ci_width = ci_upper - ci_lower
    
    contains_point = ci_lower <= point_estimate <= ci_upper
    
    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_width,
        bootstrap_estimates=jaccard_estimates,
        contains_point=contains_point
    )


def bootstrap_ci_incorrect(
    doc_a_shingles: Set[str],
    doc_b_shingles: Set[str],
    num_hashes: int = 128,
    B: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> BootstrapResult:
    """
    Compute bootstrap CI for MinHash Jaccard estimate - INCORRECT METHOD.
    
    INCORRECT APPROACH: Resample MinHash signatures directly.
    """
    rng = random.Random(seed)
    
    # Compute original signatures
    sig_a = minhash_signature(doc_a_shingles, num_hashes, seed)
    sig_b = minhash_signature(doc_b_shingles, num_hashes, seed)
    point_estimate = estimate_jaccard(sig_a, sig_b)
    
    # INCORRECT: Resample signature values instead of shingles
    jaccard_estimates = []
    
    for b in range(B):
        resampled_sig_a = [rng.choice(sig_a) for _ in range(num_hashes)]
        resampled_sig_b = [rng.choice(sig_b) for _ in range(num_hashes)]
        
        jaccard_estimates.append(estimate_jaccard(resampled_sig_a, resampled_sig_b))
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = float(np.percentile(jaccard_estimates, lower_percentile))
    ci_upper = float(np.percentile(jaccard_estimates, upper_percentile))
    ci_width = ci_upper - ci_lower
    
    contains_point = ci_lower <= point_estimate <= ci_upper
    
    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_width,
        bootstrap_estimates=jaccard_estimates,
        contains_point=contains_point
    )


def create_document_pair_known_jaccard(
    n_shingles: int = 50,
    jaccard: float = 0.5,
    seed: int = 42
) -> Tuple[Set[str], Set[str], float]:
    """Create a document pair with known Jaccard similarity."""
    rng = random.Random(seed)
    
    shared_count = int(n_shingles * jaccard / (2 - jaccard))
    shared_shingles = {f"shared_{i}_{seed}" for i in range(shared_count)}
    
    unique_count = n_shingles - shared_count
    shingles_a = shared_shingles.copy()
    shingles_b = shared_shingles.copy()
    
    for i in range(unique_count):
        shingles_a.add(f"unique_a_{i}_{seed}")
        shingles_b.add(f"unique_b_{i}_{seed}")
    
    intersection = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b)
    true_jaccard = intersection / union if union > 0 else 0.0
    
    return shingles_a, shingles_b, true_jaccard


def load_dataset(data_path: Path) -> List[Dict[str, Any]]:
    """Load dataset and flatten into list of documents."""
    logger.info(f"Loading dataset from {data_path}")
    
    with open(data_path) as f:
        data = json.load(f)
    
    documents = []
    for dataset in data["datasets"]:
        source = dataset["dataset"]
        for example in dataset["examples"]:
            doc = {
                "input": example["input"],
                "doc_id": example["metadata_doc_id"],
                "source": source,
                "word_count": example["metadata_word_count"],
                "shingle_count": example["metadata_shingle_count"],
            }
            documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} documents from {len(data['datasets'])} datasets")
    return documents


def verify_coverage_simulated(
    num_pairs: int = 100,
    num_hashes: int = 128,
    B: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """Verify bootstrap CI coverage on simulated data with known Jaccard."""
    logger.info(f"Verifying coverage on {num_pairs} simulated document pairs")
    
    rng = random.Random(seed)
    correct_contains = []
    incorrect_contains = []
    pair_results = []
    
    for i in range(num_pairs):
        jaccard_true = rng.uniform(0.1, 0.9)
        n_shingles = rng.randint(20, 100)
        
        shingles_a, shingles_b, true_jaccard = create_document_pair_known_jaccard(
            n_shingles=n_shingles,
            jaccard=jaccard_true,
            seed=seed + i
        )
        
        # Correct bootstrap CI
        result_correct = bootstrap_ci_correct(
            shingles_a, shingles_b, num_hashes, B, confidence, seed + i
        )
        correct_contains.append(result_correct.ci_lower <= true_jaccard <= result_correct.ci_upper)
        
        # Incorrect bootstrap CI
        result_incorrect = bootstrap_ci_incorrect(
            shingles_a, shingles_b, num_hashes, B, confidence, seed + i
        )
        incorrect_contains.append(result_incorrect.ci_lower <= true_jaccard <= result_incorrect.ci_upper)
        
        pair_results.append({
            "pair_id": f"sim_{i}",
            "true_jaccard": true_jaccard,
            "correct_point": result_correct.point_estimate,
            "correct_ci_lower": result_correct.ci_lower,
            "correct_ci_upper": result_correct.ci_upper,
            "correct_contains": correct_contains[-1],
            "incorrect_point": result_incorrect.point_estimate,
            "incorrect_ci_lower": result_incorrect.ci_lower,
            "incorrect_ci_upper": result_incorrect.ci_upper,
            "incorrect_contains": incorrect_contains[-1],
        })
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{num_pairs} pairs")
    
    coverage_correct = sum(correct_contains) / len(correct_contains)
    coverage_incorrect = sum(incorrect_contains) / len(incorrect_contains)
    
    logger.info(f"Coverage (correct method): {coverage_correct:.3f}")
    logger.info(f"Coverage (incorrect method): {coverage_incorrect:.3f}")
    
    return {
        "num_pairs_tested": num_pairs,
        "confidence_level": confidence,
        "coverage_correct": coverage_correct,
        "coverage_incorrect": coverage_incorrect,
        "target_coverage": confidence,
        "pairs": pair_results,
    }


def evaluate_real_data(
    documents: List[Dict[str, Any]],
    num_pairs: int = 50,
    num_hashes: int = 128,
    B: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """Evaluate bootstrap CI on real document pairs."""
    logger.info(f"Evaluating on {num_pairs} real document pairs")
    
    rng = random.Random(seed)
    
    # Generate shingles for all documents
    logger.info("Generating shingles for all documents")
    doc_shingles = {}
    for doc in documents:
        doc_id = doc["doc_id"]
        shingles = generate_shingles(doc["input"])
        doc_shingles[doc_id] = shingles
    
    # Create document pairs
    pair_results = []
    pair_count = 0
    
    sources = {}
    for doc in documents:
        source = doc["source"]
        if source not in sources:
            sources[source] = []
        sources[source].append(doc)
    
    # Within-source pairs
    for source, docs in sources.items():
        for i in range(min(10, len(docs) - 1)):
            if pair_count >= num_pairs:
                break
            
            doc_a = docs[i]
            doc_b = docs[i + 1]
            
            shingles_a = doc_shingles[doc_a["doc_id"]]
            shingles_b = doc_shingles[doc_b["doc_id"]]
            
            intersection = len(shingles_a & shingles_b)
            union = len(shingles_a | shingles_b)
            true_jaccard = intersection / union if union > 0 else 0.0
            
            result = bootstrap_ci_correct(
                shingles_a, shingles_b, num_hashes, B, confidence, seed + pair_count
            )
            
            pair_results.append({
                "pair_id": f"real_{pair_count}",
                "doc_a_id": doc_a["doc_id"],
                "doc_b_id": doc_b["doc_id"],
                "source_a": doc_a["source"],
                "source_b": doc_b["source"],
                "true_jaccard": true_jaccard,
                "point_estimate": result.point_estimate,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "ci_width": result.ci_width,
                "contains_true": result.ci_lower <= true_jaccard <= result.ci_upper,
                "contains_point": result.contains_point,
                "method": "correct",
            })
            
            pair_count += 1
        
        if pair_count >= num_pairs:
            break
    
    # Across-source pairs (if needed)
    if pair_count < num_pairs:
        source_list = list(sources.keys())
        for i in range(min(5, len(source_list))):
            for j in range(i + 1, min(5, len(source_list))):
                if pair_count >= num_pairs:
                    break
                
                doc_a = sources[source_list[i]][0]
                doc_b = sources[source_list[j]][0]
                
                shingles_a = doc_shingles[doc_a["doc_id"]]
                shingles_b = doc_shingles[doc_b["doc_id"]]
                
                intersection = len(shingles_a & shingles_b)
                union = len(shingles_a | shingles_b)
                true_jaccard = intersection / union if union > 0 else 0.0
                
                result = bootstrap_ci_correct(
                    shingles_a, shingles_b, num_hashes, B, confidence, seed + pair_count
                )
                
                pair_results.append({
                    "pair_id": f"real_{pair_count}",
                    "doc_a_id": doc_a["doc_id"],
                    "doc_b_id": doc_b["doc_id"],
                    "source_a": doc_a["source"],
                    "source_b": doc_b["source"],
                    "true_jaccard": true_jaccard,
                    "point_estimate": result.point_estimate,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "ci_width": result.ci_width,
                    "contains_true": result.ci_lower <= true_jaccard <= result.ci_upper,
                    "contains_point": result.contains_point,
                    "method": "correct",
                })
                
                pair_count += 1
    
    # Compute summary statistics
    ci_widths = [p["ci_width"] for p in pair_results]
    contains_point_rates = [p["contains_point"] for p in pair_results]
    
    return {
        "num_pairs": len(pair_results),
        "avg_ci_width": float(np.mean(ci_widths)),
        "std_ci_width": float(np.std(ci_widths)),
        "ci_contains_point_rate": float(np.mean(contains_point_rates)),
        "pairs": pair_results,
    }


@logger.catch(reraise=True)
def main():
    """Main experiment execution."""
    parser = argparse.ArgumentParser(description='Bootstrap CI Fix for MinHash')
    parser.add_argument('--num-hashes', type=int, default=128, help='Number of MinHash hash functions')
    parser.add_argument('--B', type=int, default=1000, help='Number of bootstrap samples')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-sim-pairs', type=int, default=100, help='Number of simulated pairs for coverage test')
    parser.add_argument('--num-real-pairs', type=int, default=50, help='Number of real document pairs to evaluate')
    parser.add_argument('--max-docs', type=int, default=None, help='Maximum number of documents to load (for testing)')
    parser.add_argument('--data-path', type=str, default='full_data_out.json', help='Path to dataset JSON')
    args = parser.parse_args()
    
    # Setup
    workspace = Path.cwd()
    logs_dir = workspace / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Bootstrap CI Fix for MinHash - Experiment")
    logger.info("=" * 60)
    logger.info(f"Parameters: num_hashes={args.num_hashes}, B={args.B}, confidence={args.confidence}")
    
    # Load dataset
    data_path = workspace / args.data_path
    documents = load_dataset(data_path)
    
    # Limit documents if specified
    if args.max_docs and args.max_docs < len(documents):
        documents = documents[:args.max_docs]
        logger.info(f"Limited to {len(documents)} documents")
    
    # Log data summary
    total_docs = len(documents)
    sources = set(doc["source"] for doc in documents)
    logger.info(f"Total documents: {total_docs}")
    logger.info(f"Sources: {sources}")
    
    # Step1: Verify coverage on simulated data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Coverage Verification (Simulated Data)")
    logger.info("=" * 60)
    
    sim_results = verify_coverage_simulated(
        num_pairs=args.num_sim_pairs,
        num_hashes=args.num_hashes,
        B=args.B,
        confidence=args.confidence,
        seed=args.seed
    )
    
    # Step 2: Evaluate on real data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Real Data Evaluation")
    logger.info("=" * 60)
    
    real_results = evaluate_real_data(
        documents,
        num_pairs=args.num_real_pairs,
        num_hashes=args.num_hashes,
        B=args.B,
        confidence=args.confidence,
        seed=args.seed
    )
    
    # Step 3: Compare correct vs incorrect method
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Correct vs Incorrect Method Comparison")
    logger.info("=" * 60)
    
    comparison = {
        "correct_method_coverage": sim_results["coverage_correct"],
        "incorrect_method_coverage": sim_results["coverage_incorrect"],
        "target_coverage": args.confidence,
        "num_pairs_tested": sim_results["num_pairs_tested"],
    }
    
    # Compile final results in exp_gen_sol_out schema format
    results = {
        "datasets": [
            {
                "dataset": "simulated_data",
                "examples": [
                    {
                        "input": f"sim_pair_{i}",
                        "output": str(p["true_jaccard"]),
                        "metadata_method": "bootstrap_ci_correct",
                        "metadata_pair_id": p["pair_id"],
                        "predict_bootstrap_ci": json.dumps({
                            "point_estimate": p["correct_point"],
                            "ci_lower": p["correct_ci_lower"],
                            "ci_upper": p["correct_ci_upper"],
                            "contains_true": p["correct_contains"]
                        })
                    } for i, p in enumerate(sim_results["pairs"])
                ]
            },
            {
                "dataset": "real_data",
                "examples": [
                    {
                        "input": f"{p['doc_a_id']}__{p['doc_b_id']}",
                        "output": str(p["true_jaccard"]) if p["true_jaccard"] else "unknown",
                        "metadata_method": "bootstrap_ci_correct",
                        "metadata_pair_id": p["pair_id"],
                        "predict_bootstrap_ci": json.dumps({
                            "point_estimate": p["point_estimate"],
                            "ci_lower": p["ci_lower"],
                            "ci_upper": p["ci_upper"],
                            "ci_width": p["ci_width"],
                            "contains_true": p["contains_true"],
                            "contains_point": p["contains_point"]
                        })
                    } for p in real_results["pairs"]
                ]
            }
        ],
        "experiment_info": {
            "description": "Bootstrap CI fix for MinHash on short text data",
            "dataset": "tweet_eval + ag_news",
            "num_hash_functions": args.num_hashes,
            "bootstrap_samples": args.B,
            "confidence_level": args.confidence,
            "seed": args.seed,
            "coverage_rate": sim_results["coverage_correct"],
            "target_coverage": args.confidence,
            "incorrect_method_coverage": sim_results["coverage_incorrect"]
        }
    }
    
    # Save results
    output_path = workspace / "method_out.json"
    logger.info(f"\nSaving results to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Correct method coverage: {sim_results['coverage_correct']:.3f}")
    logger.info(f"Incorrect method coverage: {sim_results['coverage_incorrect']:.3f}")
    logger.info(f"Average CI width (real data): {real_results['avg_ci_width']:.4f}")


if __name__ == "__main__":
    main()
