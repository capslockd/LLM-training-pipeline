import json
import os
import random

from datasets import Dataset


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
        for name, file_path in self.input_files.items():
            with open(file_path, "r", encoding="utf-8") as f:
                dataset_examples = [json.loads(line) for line in f if line.strip()]
                self.examples.append((name, dataset_examples))

    def mix_datasets(self, ratios=None, seed=42):
        """
        Mix datasets according to ratios.
        ratios: dict of {dataset_name: ratio}, sum <= 1.0
        """
        random.seed(seed)
        mixed = []

        if ratios is None:
            # Default: equal proportion from each dataset
            total_len = sum(len(exs) for _, exs in self.examples)
            for _, exs in self.examples:
                mixed.extend(exs)
        else:
            # Take ratio proportion from each dataset
            for name, exs in self.examples:
                r = ratios.get(name, 0)
                n_samples = int(len(exs) * r)
                mixed.extend(exs[:n_samples])

        random.shuffle(mixed)
        self.examples = mixed

    def shard_and_save(self, shard_size=5000, as_arrow=False):
        """
        Split mixed dataset into shards
        shard_size: number of examples per shard
        as_arrow: save as HuggingFace Dataset arrow files if True
        """
        total_shards = (len(self.examples) + shard_size - 1) // shard_size

        for shard_idx in range(total_shards):
            start = shard_idx * shard_size
            end = start + shard_size
            shard = self.examples[start:end]
            shard_file_jsonl = os.path.join(
                self.output_dir, f"shard_{shard_idx:03d}.jsonl"
            )

            # Save JSONL
            with open(shard_file_jsonl, "w", encoding="utf-8") as f:
                for ex in shard:
                    f.write(json.dumps(ex) + "\n")

            # Save arrow if needed
            if as_arrow:
                dataset = Dataset.from_list(shard)
                dataset.save_to_disk(
                    os.path.join(self.output_dir, f"shard_{shard_idx:03d}_arrow")
                )

        print(f"[DONE] Saved {total_shards} shards to {self.output_dir}")
