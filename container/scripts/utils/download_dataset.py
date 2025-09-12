import itertools
import json
import os

from datasets import load_dataset
from utils.dataset_monitor import DatasetCheckpoint


class RetrieveDatasets:
    def __init__(
        self, output_dir="data", checkpoint_cfg_path="config/dataset_checkpoint.yml"
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint = DatasetCheckpoint(checkpoint_cfg_path)

    def _save_subset(
        self,
        dataset_name,
        config=None,
        split="train",
        filter_name=None,
        filter_value=None,
        max_bytes=0,
        output_file=None,
        max_samples=None,
    ):
        ds = (
            load_dataset(dataset_name, config, split=split, streaming=True)
            if config
            else load_dataset(dataset_name, split=split, streaming=True)
        )

        if filter_name and filter_value:
            ds = (x for x in ds if x.get("meta", {}).get(filter_name) == filter_value)

        dataset_key = filter_value or dataset_name
        last_index = self.checkpoint.get_last_index(dataset_key)

        out = []
        total_bytes = 0
        for i, sample in enumerate(ds):
            if i < last_index:
                continue

            text = sample["text"] if isinstance(sample, dict) else str(sample)
            total_bytes += len(text.encode("utf-8"))

            if max_bytes and total_bytes > max_bytes:
                break

            out.append(json.dumps({"text": text}))

        if not output_file:
            raise ValueError("You must provide an output_file path")

        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")

        self.checkpoint.update(dataset_key, last_index + len(out), output_file)
        print(f"Appended {len(out)} samples ({total_bytes} bytes) to {output_file}")

    @classmethod
    def download_datasets(
        cls, output_dir, checkpoint_cfg_path="config/dataset_checkpoint.yml"
    ):
        retriever = cls(output_dir, checkpoint_cfg_path)

        retriever._save_subset(
            "monology/pile-uncopyrighted",
            split="train",
            filter_name="pile_set_name",
            filter_value="PubMed Abstracts",
            max_bytes=50_000_000,
            output_file=os.path.join(output_dir, "pubmed.jsonl"),
        )

        retriever._save_subset(
            "monology/pile-uncopyrighted",
            split="train",
            filter_name="pile_set_name",
            filter_value="Wikipedia (en)",
            max_bytes=20_000_000,
            output_file=os.path.join(output_dir, "wikipedia.jsonl"),
        )

        retriever._save_subset(
            "allenai/c4",
            config="en",
            split="train",
            max_bytes=50_000_000,
            output_file=os.path.join(output_dir, "c4_en.jsonl"),
        )
