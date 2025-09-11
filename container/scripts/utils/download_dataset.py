import itertools
import json
import os

from datasets import load_dataset


class RetrieveDatasets:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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
        """Download, filter, and save a subset of a dataset to JSONL."""
        # Load dataset (streaming mode)
        if config:
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)

        # Apply filter if needed
        if filter_name and filter_value:
            ds = (x for x in ds if x.get("meta", {}).get(filter_name) == filter_value)

        out = []
        total_bytes = 0

        iterator = ds if max_samples is None else itertools.islice(ds, max_samples)

        for sample in iterator:
            text = sample["text"] if isinstance(sample, dict) else str(sample)

            total_bytes += len(text.encode("utf-8"))
            if max_bytes and total_bytes > max_bytes:
                break

            out.append(json.dumps({"text": text}))

        if not output_file:
            raise ValueError("You must provide an output_file path")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(out))

        print(f"Saved {len(out)} samples ({total_bytes} bytes) to {output_file}")

    @classmethod
    def download_datasets(cls, output_dir):
        """Convenience entrypoint to download all required datasets."""
        retriever = cls(output_dir)

        retriever._save_subset(
            "monology/pile-uncopyrighted",
            split="train",
            filter_name="pile_set_name",
            filter_value="PubMed Abstracts",
            max_bytes=50_000_000,
            output_file=os.path.join(output_dir, "pubmed.jsonl"),
            max_samples=None,
        )

        retriever._save_subset(
            "monology/pile-uncopyrighted",
            split="train",
            filter_name="pile_set_name",
            filter_value="Wikipedia (en)",
            max_bytes=20_000_000,
            output_file=os.path.join(output_dir, "wikipedia.jsonl"),
            max_samples=None,
        )

        retriever._save_subset(
            "allenai/c4",
            config="en",
            split="train",
            max_bytes=50_000_000,
            output_file=os.path.join(output_dir, "c4_en.jsonl"),
            max_samples=None,
        )
