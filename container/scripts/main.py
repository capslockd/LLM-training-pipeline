# main.py
import json
import os

import yaml
from utils.data_parser import DataParser
from utils.data_prep import DatasetPreper
from utils.download_dataset import RetrieveDatasets

ENV = os.getenv("ENV", "mainpipe_nonprod")
CHECKPOINT_CONFIG_PATH = f"config/dataset_{ENV}_checkpoint.yml"


def load_config():
    config_path = f"config/{ENV}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()

    INPUT_DIR = config["paths"]["input_dir"]
    OUTPUT_DIR_PROCESSED = config["paths"]["processed_dir"]
    OUTPUT_DIR_SHARDS = config["paths"]["shards_dir"]

    RetrieveDatasets.download_datasets(
        output_dir=INPUT_DIR, checkpoint_cfg_path=CHECKPOINT_CONFIG_PATH
    )
    parser = DataParser(config)
    os.makedirs(OUTPUT_DIR_PROCESSED, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SHARDS, exist_ok=True)
    dataset_files = config.get("datasets", [])

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
    preparer.debug_dataset(config["debug"])
    preparer.inspect_dataset(config["inspect"])
    preparer.shard_and_save(
        shard_size=config["sharding"]["shard_size"],
        as_arrow=config["sharding"]["as_arrow"],
    )

    print("All datasets processed, mixed, inspected, and sharded successfully!")
