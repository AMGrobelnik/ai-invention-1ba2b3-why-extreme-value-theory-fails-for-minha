#!/usr/bin/env python3
"""Analyze text lengths and download samples from candidate datasets."""

from loguru import logger
from pathlib import Path
import json
import sys
import re
from collections import Counter

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def analyze_text_lengths(texts, dataset_name):
    """Analyze word counts and shingle counts for a list of texts."""
    word_counts = []
    shingle_counts = []
    
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        
        # Word count (split on whitespace, filter empty)
        words = [w for w in text.split() if w]
        word_count = len(words)
        
        # k=3 character shingles (unique)
        if len(text) >= 3:
            shingles = set()
            for i in range(len(text) - 2):
                shingle = text[i:i+3]
                shingles.add(shingle)
            shingle_count = len(shingles)
        else:
            shingle_count = 0
        
        word_counts.append(word_count)
        shingle_counts.append(shingle_count)
    
    # Statistics
    word_counter = Counter(word_counts)
    total = len(word_counts)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Total documents: {total}")
    logger.info(f"\nWord count distribution:")
    logger.info(f"  Min: {min(word_counts)}")
    logger.info(f"  Max: {max(word_counts)}")
    logger.info(f"  Mean: {sum(word_counts)/len(word_counts):.1f}")
    logger.info(f"  Median: {sorted(word_counts)[len(word_counts)//2]}")
    
    # Percentiles
    sorted_wc = sorted(word_counts)
    for p in [10, 25, 50, 75, 80, 90, 95]:
        idx = int(total * p / 100)
        logger.info(f"  {p}th percentile: {sorted_wc[idx]}")
    
    # Count in range 10-100
    in_range = sum(1 for wc in word_counts if 10 <= wc <= 100)
    logger.info(f"\nDocuments with 10-100 words: {in_range} ({100*in_range/total:.1f}%)")
    
    # Shingle count stats
    logger.info(f"\nShingle count (k=3) distribution:")
    logger.info(f"  Min: {min(shingle_counts)}")
    logger.info(f"  Max: {max(shingle_counts)}")
    logger.info(f"  Mean: {sum(shingle_counts)/len(shingle_counts):.1f}")
    
    return {
        "total_docs": total,
        "word_counts": word_counts,
        "shingle_counts": shingle_counts,
        "in_range_10_100": in_range,
        "percent_in_range": 100*in_range/total
    }

@logger.catch(reraise=True)
def main():
    import datasets
    
    # Candidate datasets to analyze
    candidates = [
        ("cardiffnlp/tweet_eval", "emotion", "train", "text"),
        ("ucirvine/sms_spam", "plain_text", "train", "sms"),
        ("bguzzo2k/nyt_100y_news_headlines", "default", "train", "headline"),
    ]
    
    results = {}
    
    for dataset_id, config, split, text_field in candidates:
        logger.info(f"\nLoading {dataset_id} (config={config}, split={split})...")
        
        try:
            # Load dataset
            dataset = datasets.load_dataset(
                dataset_id, 
                config, 
                split=split,
                streaming=True
            )
            
            # Get first 1000 samples
            texts = []
            for i, row in enumerate(dataset):
                if i >= 1000:
                    break
                if text_field in row:
                    texts.append(row[text_field])
            
            # Analyze
            stats = analyze_text_lengths(texts, dataset_id)
            results[dataset_id] = stats
            
        except Exception as e:
            logger.error(f"Failed to load {dataset_id}: {e}")
    
    # Save results
    output_path = Path("temp/dataset_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\nSaved analysis to {output_path}")

if __name__ == "__main__":
    main()
