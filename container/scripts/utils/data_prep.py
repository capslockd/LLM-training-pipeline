import json
import logging
import os
import random

from datasets import Dataset

logger = logging.getLogger(__name__)


class DatasetPreper:
    def __init__(self, input_files, output_dir="scripts/processed_shards"):
        """
        input_files: dict of {dataset_name: file_path}
        output_dir: where shards will be saved
        """
        self.input_files = input_files
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.examples = []

    def load_datasets(self):
        """Load JSONL datasets into memory"""
        loaded = {}
        for name, file_path in self.input_files.items():
            with open(file_path, "r", encoding="utf-8") as f:
                dataset_examples = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(dataset_examples)} examples from {name}")
                # Save inspection file
                inspect_file = os.path.join(self.output_dir, f"{name}_inspect.jsonl")
                with open(inspect_file, "w") as insp:
                    for ex in dataset_examples[:5]:
                        insp.write(json.dumps(ex) + "\n")
                logger.info(f"[INSPECT] Saved 5 samples for {name} -> {inspect_file}")
                loaded[name] = dataset_examples
        self.loaded_datasets = loaded

    def mix_datasets(self, ratios=None, seed=42):
        """
        Mix datasets according to ratios.
        ratios: dict of {dataset_name: ratio}, sum <= 1.0
        """
        random.seed(seed)
        mixed = []

        if ratios is None:
            for name, exs in self.loaded_datasets.items():
                mixed.extend(exs)
        else:
            for name, exs in self.loaded_datasets.items():
                r = ratios.get(name, 0)
                n_samples = max(1, int(len(exs) * r))
                mixed.extend(exs[:n_samples])
                logger.info(f"Using {n_samples} samples from {name} (ratio {r})")

        random.shuffle(mixed)
        self.examples = mixed
        logger.info(f"Mixed dataset size: {len(self.examples)} examples")

    def shard_and_save(self, shard_size=5000, as_arrow=False):
        """
        Split mixed dataset into multiple shards
        shard_size: number of examples per shard
        as_arrow: save as HuggingFace Dataset arrow files if True
        """
        total_shards = (len(self.examples) + shard_size - 1) // shard_size

        for shard_idx in range(total_shards):
            start = shard_idx * shard_size
            end = start + shard_size
            shard = self.examples[start:end]

            if not shard:
                continue

            shard_file_jsonl = os.path.join(
                self.output_dir, f"shard_{shard_idx:03d}.jsonl"
            )
            with open(shard_file_jsonl, "w", encoding="utf-8") as f:
                for ex in shard:
                    f.write(json.dumps(ex) + "\n")

            # Write metadata
            metadata_file = os.path.join(
                self.output_dir, f"shard_{shard_idx:03d}_meta.json"
            )
            with open(metadata_file, "w") as meta:
                json.dump(
                    {
                        "shard_idx": shard_idx,
                        "num_examples": len(shard),
                        "example_sources": list(
                            {ex.get("source", "unknown") for ex in shard}
                        ),
                    },
                    meta,
                )

            if as_arrow:
                dataset = Dataset.from_list(shard)
                dataset.save_to_disk(
                    os.path.join(self.output_dir, f"shard_{shard_idx:03d}_arrow")
                )

            logger.info(f"[DONE] Saved shard {shard_idx:03d} ({len(shard)} examples)")

        logger.info(f"Saved {total_shards} shards to {self.output_dir}")
