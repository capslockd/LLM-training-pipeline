import json
import os

from datasets import load_dataset


def save_subset(
    dataset_name,
    config=None,
    split="train",
    filter_name=None,
    filter_value=None,
    max_bytes=0,
    output_file="out.jsonl",
    max_samples=None,
):
    """
    Save a subset of a dataset to a JSONL file, using streaming.
    Stops early when max_samples or max_bytes is reached.
    """

    if config:
        ds = load_dataset(dataset_name, config, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=True)

    if filter_name and filter_value:
        ds = ds.filter(lambda x: x.get("meta", {}).get(filter_name) == filter_value)

    total_bytes, collected = 0, 0

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in ds:
            if max_samples and collected >= max_samples:
                break

            text = sample.get("text", "") if isinstance(sample, dict) else str(sample)
            encoded = text.encode("utf-8")
            if max_bytes and (total_bytes + len(encoded) > max_bytes):
                break

            f.write(json.dumps({"text": text}) + "\n")
            total_bytes += len(encoded)
            collected += 1

    print(f"✅ Saved {collected} samples ({total_bytes} bytes) to {output_file}")


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # PubMed
    save_subset(
        "monology/pile-uncopyrighted",
        split="train",
        filter_name="pile_set_name",
        filter_value="PubMed Abstracts",
        max_bytes=50_000_000,
        output_file=os.path.join(OUTPUT_DIR, "pubmed.jsonl"),
        max_samples=None,
    )

    # Wikipedia
    save_subset(
        "monology/pile-uncopyrighted",
        split="train",
        filter_name="pile_set_name",
        filter_value="Wikipedia (en)",
        max_bytes=20_000_000,
        output_file=os.path.join(OUTPUT_DIR, "wikipedia.jsonl"),
        max_samples=None,
    )

    # C4
    save_subset(
        "allenai/c4",
        config="en",
        split="train",
        max_bytes=50_000_000,
        output_file=os.path.join(OUTPUT_DIR, "c4_en.jsonl"),
        max_samples=None,
    )
