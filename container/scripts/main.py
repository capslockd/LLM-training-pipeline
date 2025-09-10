import json
import os
import random

from utils.data_parser import DataParser
from utils.data_prep import DatasetPreper

# Paths
INPUT_DIR = "data"  # raw downloaded JSONL files
OUTPUT_DIR_PROCESSED = "processed"  # intermediate processed/tokenized files
OUTPUT_DIR_SHARDS = "processed_shards"  # final shards
os.makedirs(OUTPUT_DIR_PROCESSED, exist_ok=True)
os.makedirs(OUTPUT_DIR_SHARDS, exist_ok=True)

# Dataset files
dataset_files = [
    "pubmed.jsonl",
    # "github.jsonl",
    "wikipedia.jsonl",
    "c4_en.jsonl",
]

# Mixing ratios (can be adjusted)
ratios = {
    "pubmed": 0.25,
    # "github": 0.25,
    "wikipedia": 0.25,
    "c4": 0.25,
}

SHARD_SIZE = 5000  # examples per shard


def preprocess_file(file_name):
    input_path = os.path.join(INPUT_DIR, file_name)
    processed_data = []

    # Load raw text
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            # Clean & normalize
            cleaned = DataParser.clean_text(text)
            normalized = DataParser.normalize_text(cleaned)
            processed_data.append(normalized)

    # Tokenize
    tokenized = DataParser.tokenize_text(processed_data)

    # Save processed/tokenized examples for intermediate storage
    processed_file_path = os.path.join(
        OUTPUT_DIR_PROCESSED, f"{file_name.split('.')[0]}_processed.jsonl"
    )
    with open(processed_file_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(processed_data):
            f.write(
                json.dumps(
                    {"text": text, "input_ids": tokenized["input_ids"][i].tolist()}
                )
                + "\n"
            )

    print(f"[DONE] {file_name} -> {processed_file_path}, {len(processed_data)} samples")
    return processed_file_path


if __name__ == "__main__":
    # Step 1-3: Clean, normalize, tokenize
    processed_files = {}
    for file_name in dataset_files:
        processed_path = preprocess_file(file_name)
        key = file_name.split(".")[0]
        processed_files[key] = processed_path

    # Step 4-5: Mix datasets and shard into training-ready files
    preparer = DatasetPreper(processed_files, output_dir=OUTPUT_DIR_SHARDS)
    preparer.load_datasets()
    preparer.mix_datasets(ratios=ratios)
    preparer.shard_and_save(shard_size=SHARD_SIZE, as_arrow=False)

    print("All datasets processed, mixed, and sharded successfully!")
