#!/usr/bin/env python3
"""Download and analyze multiple datasets for EVT-MinHash evaluation."""

from loguru import logger
from pathlib import Path
import json
import sys
import datasets

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def analyze_dataset(dataset_id, config, split, text_field, max_samples=10000):
    """Download and analyze a dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing: {dataset_id} (config={config}, split={split})")
    
    try:
        # Load dataset
        dataset = datasets.load_dataset(
            dataset_id, 
            config, 
            split=split,
            streaming=False  # Need random access for sampling
        )
        
        total = len(dataset)
        logger.info(f"Total examples: {total}")
        
        # Sample up to max_samples
        if total > max_samples:
            indices = range(max_samples)
        else:
            indices = range(total)
        
        texts = []
        for i in indices:
            if text_field in dataset[i]:
                text = dataset[i][text_field]
                if text and isinstance(text, str):
                    texts.append(text)
        
        logger.info(f"Analyzing {len(texts)} samples...")
        
        # Analyze text lengths
        word_counts = []
        shingle_counts = []
        
        for text in texts:
            words = [w for w in text.split() if w]
            word_count = len(words)
            
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
        in_range = sum(1 for wc in word_counts if 10 <= wc <= 100)
        percent_in_range = 100 * in_range / len(word_counts) if word_counts else 0
        
        sorted_wc = sorted(word_counts)
        n = len(sorted_wc)
        
        logger.info(f"\nWord count distribution:")
        logger.info(f"  Min: {min(word_counts)}")
        logger.info(f"  Max: {max(word_counts)}")
        logger.info(f"  Mean: {sum(word_counts)/n:.1f}")
        logger.info(f"  Median: {sorted_wc[n//2]}")
        logger.info(f"  80th percentile: {sorted_wc[int(n*0.8)]}")
        
        logger.info(f"\nDocuments with 10-100 words: {in_range} ({percent_in_range:.1f}%)")
        
        logger.info(f"\nShingle count (k=3) distribution:")
        logger.info(f"  Min: {min(shingle_counts)}")
        logger.info(f"  Max: {max(shingle_counts)}")
        logger.info(f"  Mean: {sum(shingle_counts)/len(shingle_counts):.1f}")
        
        # Save standardized samples
        output_dir = Path("temp/datasets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        standardized = []
        for i, text in enumerate(texts[:1000]):  # Save first 1000
            words = [w for w in text.split() if w]
            word_count = len(words)
            
            if len(text) >= 3:
                shingles = set()
                for j in range(len(text) - 2):
                    shingles.add(text[j:j+3])
                shingle_count = len(shingles)
            else:
                shingle_count = 0
            
            standardized.append({
                "doc_id": f"{dataset_id.replace('/', '_')}_{i}",
                "text": text,
                "source": dataset_id,
                "length": {
                    "words": word_count,
                    "shingles_k3": shingle_count
                }
            })
        
        # Save full, mini, preview versions
        dataset_name = dataset_id.replace('/', '_')
        
        # Full (all samples)
        full_path = output_dir / f"full_{dataset_name}_{config}_{split}.json"
        with open(full_path, 'w') as f:
            json.dump(standardized, f, indent=2)
        logger.info(f"Saved full dataset: {full_path}")
        
        # Mini (first 100)
        mini_path = output_dir / f"mini_{dataset_name}_{config}_{split}.json"
        with open(mini_path, 'w') as f:
            json.dump(standardized[:100], f, indent=2)
        
        # Preview (first 3, truncated text)
        preview = []
        for item in standardized[:3]:
            preview_item = item.copy()
            if len(preview_item['text']) > 100:
                preview_item['text'] = preview_item['text'][:100] + "..."
            preview.append(preview_item)
        
        preview_path = output_dir / f"preview_{dataset_name}_{config}_{split}.json"
        with open(preview_path, 'w') as f:
            json.dump(preview, f, indent=2)
        
        return {
            "dataset_id": dataset_id,
            "config": config,
            "split": split,
            "total_examples": total,
            "analyzed_samples": len(texts),
            "in_range_10_100": in_range,
            "percent_in_range": percent_in_range,
            "files": {
                "full": str(full_path),
                "mini": str(mini_path),
                "preview": str(preview_path)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze {dataset_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

@logger.catch(reraise=True)
def main():
    # Datasets to analyze
    candidates = [
        # (dataset_id, config, split, text_field)
        ("cardiffnlp/tweet_eval", "sentiment", "train", "text"),  # 45K examples
        ("cardiffnlp/tweet_eval", "emoji", "train", "text"),  # 45K examples
        ("ucirvine/sms_spam", "plain_text", "train", "sms"),  # 5.5K examples
        ("Yelp/yelp_review_full", None, "train", "text"),  # 650K examples
    ]
    
    results = []
    
    for dataset_id, config, split, text_field in candidates:
        result = analyze_dataset(dataset_id, config, split, text_field)
        if result:
            results.append(result)
    
    # Save summary
    summary_path = Path("temp/dataset_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary saved to: {summary_path}")
    
    # Print recommendation
    logger.info(f"\n{'='*60}")
    logger.info("RECOMMENDATIONS:")
    for r in results:
        if r and r['percent_in_range'] > 50:
            logger.info(f"  ✓ {r['dataset_id']} ({r['config']}): {r['percent_in_range']:.1f}% in range, {r['total_examples']} total")

if __name__ == "__main__":
    main()
