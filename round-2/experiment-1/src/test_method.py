#!/usr/bin/env python3
"""
Quick test script to verify the experiment works with smaller parameters.
"""

from loguru import logger
from pathlib import Path
import json
import sys

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

from method import (
    generate_shingles, 
    minhash_signature, 
    estimate_jaccard,
    bootstrap_ci_correct,
    bootstrap_ci_incorrect,
    create_document_pair_known_jaccard,
    load_dataset,
    verify_coverage_simulated,
    evaluate_real_data
)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

@logger.catch(reraise=True)
def main():
    """Run a quick test with small parameters."""
    workspace = Path.cwd()
    
    logger.info("Running quick test with small parameters...")
    
    # Test 1: Verify coverage with small simulated dataset
    logger.info("\n=== TEST 1: Small-scale coverage verification ===")
    sim_results = verify_coverage_simulated(
        num_pairs=10,      # Only 10 pairs
        num_hashes=32,     # Fewer hash functions
        B=100,             # Fewer bootstrap samples
        confidence=0.95,
        seed=42
    )
    logger.info(f"Coverage (correct): {sim_results['coverage_correct']:.3f}")
    logger.info(f"Coverage (incorrect): {sim_results['coverage_incorrect']:.3f}")
    
    # Test 2: Load real data (limited)
    logger.info("\n=== TEST 2: Load real data (limited) ===")
    data_path = workspace / "full_data_out.json"
    
    # Load only first 20 documents
    with open(data_path) as f:
        data = json.load(f)
    
    documents = []
    count = 0
    for dataset in data["datasets"]:
        for example in dataset["examples"]:
            doc = {
                "input": example["input"],
                "doc_id": example["metadata_doc_id"],
                "source": dataset["dataset"],
                "word_count": example["metadata_word_count"],
                "shingle_count": example["metadata_shingle_count"],
            }
            documents.append(doc)
            count += 1
            if count >= 20:
                break
        if count >= 20:
            break
    
    logger.info(f"Loaded {len(documents)} documents for testing")
    
    # Test 3: Evaluate on real data (small scale)
    logger.info("\n=== TEST 3: Real data evaluation (small scale) ===")
    real_results = evaluate_real_data(
        documents,
        num_pairs=5,
        num_hashes=32,
        B=100,
        confidence=0.95,
        seed=42
    )
    logger.info(f"Number of pairs evaluated: {real_results['num_pairs']}")
    logger.info(f"Average CI width: {real_results['avg_ci_width']:.4f}")
    logger.info(f"CI contains point rate: {real_results['ci_contains_point_rate']:.3f}")
    
    # Test 4: Save mini results
    logger.info("\n=== TEST 4: Save results ===")
    results = {
        "experiment_info": {
            "description": "TEST RUN - Bootstrap CI fix for MinHash",
            "test": True,
        },
        "simulated_data_results": {
            "coverage_rate": sim_results["coverage_correct"],
            "num_pairs_tested": sim_results["num_pairs_tested"],
        },
        "real_data_results": {
            "num_pairs": real_results["num_pairs"],
            "avg_ci_width": real_results["avg_ci_width"],
        },
    }
    
    output_path = workspace / "method_out_test.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to: {output_path}")
    logger.info("\n=== ALL TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
