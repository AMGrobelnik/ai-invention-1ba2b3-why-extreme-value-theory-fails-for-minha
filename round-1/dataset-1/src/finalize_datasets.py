#!/usr/bin/env python3
"""Download and process final datasets for EVT-MinHash evaluation."""

from loguru import logger
from pathlib import Path
import json
import sys
import datasets

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def download_and_process(dataset_id, config, split, text_field, max_docs=20000):
    """Download and process a dataset into standardized format."""
    logger.info(f"Processing {dataset_id}...")
    
    try:
        # Load dataset
        if config:
            ds = datasets.load_dataset(dataset_id, config, split=split, streaming=False)
        else:
            ds = datasets.load_dataset(dataset_id, split=split, streaming=False)
        
        total = len(ds)
        logger.info(f"  Total examples: {total}")
        
        # Process documents
        standardized = []
        for i, row in enumerate(ds):
            if i >= max_docs:
                break
            
            if text_field not in row:
                continue
                
            text = row[text_field]
            if not text or not isinstance(text, str):
                continue
            
            # Count words
            words = [w for w in text.split() if w]
            word_count = len(words)
            
            # Skip if not in range (10-100 words)
            if not (10 <= word_count <= 100):
                continue
            
            # Count k=3 shingles
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
        
        logger.info(f"  Kept {len(standardized)} documents in range")
        
        # Save
        output_dir = Path("temp/datasets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_name = dataset_id.replace('/', '_')
        if config:
            dataset_name = f"{dataset_name}_{config}"
        
        # Full
        full_path = output_dir / f"full_{dataset_name}_{split}.json"
        with open(full_path, 'w') as f:
            json.dump(standardized, f, indent=2)
        
        # Mini (first 100)
        mini_path = output_dir / f"mini_{dataset_name}_{split}.json"
        with open(mini_path, 'w') as f:
            json.dump(standardized[:100], f, indent=2)
        
        # Preview (first 3, truncated)
        preview = []
        for item in standardized[:3]:
            preview_item = item.copy()
            if len(preview_item['text']) > 100:
                preview_item['text'] = preview_item['text'][:100] + "..."
            preview.append(preview_item)
        
        preview_path = output_dir / f"preview_{dataset_name}_{split}.json"
        with open(preview_path, 'w') as f:
            json.dump(preview, f, indent=2)
        
        logger.info(f"  Saved: {full_path.name}")
        
        return {
            "dataset_id": dataset_id,
            "config": config,
            "split": split,
            "total_docs": len(standardized),
            "files": {
                "full": str(full_path),
                "mini": str(mini_path),
                "preview": str(preview_path)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

@logger.catch(reraise=True)
def main():
    # Process final 3 datasets
    datasets_to_process = [
        ("fancyzhx/ag_news", None, "train", "text", 20000),
        ("cornell-movie-review-data/rotten_tomatoes", None, "train", "text", 8530),
    ]
    
    results = []
    
    # Process AG News and Rotten Tomatoes
    for dataset_id, config, split, text_field, max_docs in datasets_to_process:
        result = download_and_process(dataset_id, config, split, text_field, max_docs)
        if result:
            results.append(result)
    
    # Now merge with previously downloaded datasets
    logger.info(f"\n{'='*60}")
    logger.info("Merging all datasets...")
    
    # Load previous datasets
    prev_datasets = [
        "temp/datasets/full_cardiffnlp_tweet_eval_sentiment_train.json",
        "temp/datasets/full_cardiffnlp_tweet_eval_emoji_train.json",
    ]
    
    all_docs = []
    
    # Add from previous datasets
    for path in prev_datasets:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                docs = json.load(f)
            all_docs.extend(docs)
            logger.info(f"Added {len(docs)} from {p.name}")
    
    # Add from new datasets
    for result in results:
        full_path = result['files']['full']
        with open(full_path) as f:
            docs = json.load(f)
        all_docs.extend(docs)
        logger.info(f"Added {len(docs)} from {result['dataset_id']}")
    
    # Create final data_out.json
    output = {
        "metadata": {
            "total_docs": len(all_docs),
            "datasets": ["cardiffnlp/tweet_eval (sentiment)", "cardiffnlp/tweet_eval (emoji)", 
                         "fancyzhx/ag_news", "cornell-movie-review-data/rotten_tomatoes"],
            "avg_words": sum(d['length']['words'] for d in all_docs) / len(all_docs) if all_docs else 0
        },
        "documents": all_docs
    }
    
    output_path = Path("data_out.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL OUTPUT: {output_path}")
    logger.info(f"Total documents: {len(all_docs)}")
    logger.info(f"Total size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Create preview
    preview = {"metadata": output['metadata'], "documents": []}
    for doc in all_docs[:3]:
        preview_doc = doc.copy()
        if len(preview_doc['text']) > 100:
            preview_doc['text'] = preview_doc['text'][:100] + "..."
        preview['documents'].append(preview_doc)
    
    preview_path = Path("preview_data_out.json")
    with open(preview_path, 'w') as f:
        json.dump(preview, f, indent=2)
    
    logger.info(f"Preview saved: {preview_path}")

if __name__ == "__main__":
    main()
