# utils/data_prep.py
import hashlib
import json
import os
import random
import re
from collections import Counter

import matplotlib.pyplot as plt
from datasets import Dataset

try:
    from langdetect import detect
except ImportError:
    detect = None


class DatasetPreper:
    def __init__(self, input_files, output_dir="scripts/processed_shards"):
        self.input_files = input_files
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.examples = []

    def load_datasets(self):
        loaded = []
        for name, file_path in self.input_files.items():
            with open(file_path, "r", encoding="utf-8") as f:
                dataset_examples = [json.loads(line) for line in f if line.strip()]
                loaded.append((name, dataset_examples))
        self.loaded_datasets = dict(loaded)

    def mix_datasets(self, ratios=None, seed=42):
        random.seed(seed)
        mixed = []
        if ratios is None:
            for _, exs in self.loaded_datasets.items():
                mixed.extend(exs)
        else:
            for name, exs in self.loaded_datasets.items():
                r = ratios.get(name, 0)
                n_samples = max(1, int(len(exs) * r))
                mixed.extend(exs[:n_samples])
        random.shuffle(mixed)
        self.examples = mixed

    def debug_dataset(self, config):
        """Quick debug prints for sample inspection"""
        if not config.get("enabled", False):
            return
        n = config.get("inspect_samples", 5)
        verbose = config.get("verbose", False)

        print("\n[DEBUG] Showing first few samples:")
        for i, ex in enumerate(self.examples[:n]):
            print(f"--- Sample {i+1} ---")
            print("Text:", ex.get("text", "")[:200], "...")
            if verbose:
                print("Input IDs:", ex.get("input_ids", [])[:20])
        print("[DEBUG] Done.\n")

    def inspect_dataset(self, config):
        """Full inspectability analysis (lengths, langs, PII, etc.)"""
        if not config.get("enabled", False):
            return

        save_dir = config.get("output_dir", "inspect_reports")
        os.makedirs(save_dir, exist_ok=True)

        lengths = [len(ex["text"].split()) for ex in self.examples]
        plt.hist(lengths, bins=50)
        plt.title("Text Length Histogram")
        plt.xlabel("Tokens")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(save_dir, "length_hist.png"))
        plt.close()

        report = {
            "num_samples": len(self.examples),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        }

        if config.get("detect_language", False) and detect:
            langs = [detect(ex["text"]) for ex in self.examples[:500]]
            report["lang_counts"] = dict(Counter(langs))

        if config.get("check_duplicates", False):
            hashes = [
                hashlib.md5(ex["text"].encode()).hexdigest() for ex in self.examples
            ]
            report["duplicate_count"] = len(hashes) - len(set(hashes))

        if config.get("check_pii", False):
            pii_hits = sum(
                bool(re.search(r"\b[\w.-]+?@\w+?\.\w+?\b", ex["text"]))
                for ex in self.examples
            )
            report["pii_hits"] = pii_hits

        with open(os.path.join(save_dir, "inspect_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        print(f"[INSPECT] Report saved to {save_dir}")

    def shard_and_save(self, shard_size=5000, as_arrow=False):
        """
        Split mixed dataset into multiple shards
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

            if as_arrow:
                dataset = Dataset.from_list(shard)
                dataset.save_to_disk(
                    os.path.join(self.output_dir, f"shard_{shard_idx:03d}_arrow")
                )

        print(f"[DONE] Saved {total_shards} shards to {self.output_dir}")
