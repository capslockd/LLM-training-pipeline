import json
import logging
import os

from utils.data_parser import DataParser
from utils.data_prep import DatasetPreper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
INPUT_DIR = "data"  # raw downloaded JSONL files
OUTPUT_DIR_PROCESSED = "processed"  # intermediate processed/tokenized files
OUTPUT_DIR_SHARDS = "processed_shards"  # final shards
os.makedirs(OUTPUT_DIR_PROCESSED, exist_ok=True)
os.makedirs(OUTPUT_DIR_SHARDS, exist_ok=True)

# Dataset files
dataset_files = [
    "pubmed.jsonl",
    "wikipedia.jsonl",
    "c4_en.jsonl",
]

# Mixing ratios
ratios = {
    "pubmed": 0.25,
    "wikipedia": 0.25,
    "c4": 0.25,
}

SHARD_SIZE = 1000


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
            cleaned = DataParser.clean_text(text)
            normalized = DataParser.normalize_text(cleaned)
            processed_data.append(normalized)

    assert processed_data, f"No valid examples found in {file_name}"

    tokenized = DataParser.tokenize_text(processed_data)

    processed_file_path = os.path.join(
        OUTPUT_DIR_PROCESSED, f"{file_name.split('.')[0]}_processed.jsonl"
    )
    with open(processed_file_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(processed_data):
            f.write(
                json.dumps(
                    {
                        "text": text,
                        "input_ids": tokenized["input_ids"][i].tolist(),
                        "source": file_name.split(".")[0],
                    }
                )
                + "\n"
            )

    # Save inspection file
    inspect_file = processed_file_path.replace(".jsonl", "_inspect.jsonl")
    with open(inspect_file, "w", encoding="utf-8") as f:
        for i in range(min(5, len(processed_data))):
            f.write(
                json.dumps(
                    {
                        "text": processed_data[i],
                        "input_ids": tokenized["input_ids"][i].tolist(),
                        "source": file_name.split(".")[0],
                    }
                )
                + "\n"
            )
    logger.info(f"[INSPECT] Saved 5 examples for {file_name} -> {inspect_file}")

    logger.info(
        f"[DONE] {file_name} -> {processed_file_path}, {len(processed_data)} samples"
    )
    return processed_file_path


if __name__ == "__main__":
    processed_files = {}
    for file_name in dataset_files:
        processed_path = preprocess_file(file_name)
        key = file_name.split(".")[0]
        processed_files[key] = processed_path

    preparer = DatasetPreper(processed_files, output_dir=OUTPUT_DIR_SHARDS)
    preparer.load_datasets()
    preparer.mix_datasets(ratios=ratios)
    preparer.shard_and_save(shard_size=SHARD_SIZE, as_arrow=False)

    logger.info("All datasets processed, mixed, and sharded successfully!")
