#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru",
# ]
# ///

"""
Convert collected short text datasets to exp_sel_data_out.json schema.

For near-duplicate detection evaluation:
- input: document text
- output: doc_id (document identifier)
- metadata_source: dataset source
- metadata_word_count: number of words
- metadata_shingle_count: number of k=3 shingles
"""

from loguru import logger
from pathlib import Path
import json
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")


@logger.catch(reraise=True)
def main():
    # Load the collected datasets from data_out.json
    data_out_path = Path("data_out.json")
    
    if not data_out_path.exists():
        logger.error(f"{data_out_path} not found. Run dataset collection first.")
        sys.exit(1)
    
    logger.info(f"Loading {data_out_path}")
    with open(data_out_path) as f:
        data = json.load(f)
    
    # Group documents by source dataset
    datasets_dict = {}
    for doc in data["documents"]:
        source = doc["source"]
        if source not in datasets_dict:
            datasets_dict[source] = []
        
        # Create example in required format
        example = {
            "input": doc["text"],
            "output": doc["doc_id"],  # Use doc_id as output identifier
            "metadata_source": source,
            "metadata_word_count": doc["length"]["words"],
            "metadata_shingle_count": doc["length"]["shingles_k3"],
            "metadata_doc_id": doc["doc_id"],
        }
        datasets_dict[source].append(example)
    
    # Convert to required schema format
    datasets_list = []
    for source, examples in datasets_dict.items():
        datasets_list.append({
            "dataset": source,
            "examples": examples
        })
    
    # Create output
    output = {
        "metadata": {
            "total_datasets": len(datasets_list),
            "total_examples": sum(len(d["examples"]) for d in datasets_list),
            "source_datasets": list(datasets_dict.keys()),
        },
        "datasets": datasets_list
    }
    
    # Save to full_data_out.json
    output_path = Path("full_data_out.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved {output_path}")
    logger.info(f"  Total datasets: {len(datasets_list)}")
    for ds in datasets_list:
        logger.info(f"  - {ds['dataset']}: {len(ds['examples'])} examples")
    
    # Create preview (first 3 examples from first dataset)
    preview = {"metadata": output["metadata"], "datasets": []}
    for ds in datasets_list:
        preview_ds = {"dataset": ds["dataset"], "examples": []}
        for ex in ds["examples"][:3]:
            preview_ex = ex.copy()
            if len(preview_ex["input"]) > 100:
                preview_ex["input"] = preview_ex["input"][:100] + "..."
            preview_ds["examples"].append(preview_ex)
        preview["datasets"].append(preview_ds)
    
    preview_path = Path("preview_full_data_out.json")
    with open(preview_path, 'w') as f:
        json.dump(preview, f, indent=2)
    
    logger.info(f"Saved preview: {preview_path}")


if __name__ == "__main__":
    main()
