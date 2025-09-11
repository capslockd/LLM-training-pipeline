# main.py
import json
import os

import yaml
from utils.data_parser import DataParser
from utils.data_prep import DatasetPreper


def load_config():
    env = os.getenv("ENV", "mainpipe_nonprod")
    config_path = f"config/{env}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    parser = DataParser(config)
    INPUT_DIR = "data"
    OUTPUT_DIR_PROCESSED = "processed"
    OUTPUT_DIR_SHARDS = "processed_shards"
    os.makedirs(OUTPUT_DIR_PROCESSED, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SHARDS, exist_ok=True)

    dataset_files = ["pubmed.jsonl", "wikipedia.jsonl", "c4_en.jsonl"]

    processed_files = {}
    for file_name in dataset_files:
        input_path = os.path.join(INPUT_DIR, file_name)
        processed_data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                text = json.loads(line)["text"]
                cleaned = parser.clean_text(text)
                normalized = parser.normalize_text(cleaned)
                processed_data.append(normalized)

        tokenized = parser.tokenize_text(processed_data)

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
        processed_files[file_name.split(".")[0]] = processed_file_path

    preparer = DatasetPreper(processed_files, output_dir=OUTPUT_DIR_SHARDS)
    preparer.load_datasets()
    preparer.mix_datasets(ratios=config["mixing"]["ratios"])
    preparer.debug_dataset(config["debug"])  # 👈 Debug (lightweight)
    preparer.inspect_dataset(config["inspect"])  # 👈 Inspect (deep)
    preparer.shard_and_save(
        shard_size=config["sharding"]["shard_size"],
        as_arrow=config["sharding"]["as_arrow"],
    )

    print("✅ All datasets processed, mixed, inspected, and sharded successfully!")
