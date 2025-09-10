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

    # Load dataset
    if config:
        ds = load_dataset(dataset_name, config, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=True)

    # Apply filter if required
    if filter_name and filter_value:
        ds = (x for x in ds if x.get("meta", {}).get(filter_name) == filter_value)

    out = []
    total_bytes = 0
    collected = 0

    for sample in ds:
        # Stop if we reached max_samples
        if max_samples and collected >= max_samples:
            break

        # Extract text
        text = sample.get("text", "") if isinstance(sample, dict) else str(sample)

        # Stop if we reached max_bytes
        if max_bytes and (total_bytes + len(text.encode("utf-8")) > max_bytes):
            break

        out.append(json.dumps({"text": text}))
        total_bytes += len(text.encode("utf-8"))
        collected += 1

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Saved {collected} samples ({total_bytes} bytes) to {output_file}")


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
    # OUTPUT_DIR = "/usr/scripts/data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Small test runs for speed
    save_subset(
        "monology/pile-uncopyrighted",
        split="train",
        filter_name="pile_set_name",
        filter_value="PubMed Abstracts",
        max_bytes=50_000,
        output_file=os.path.join(OUTPUT_DIR, "pubmed.jsonl"),
        max_samples=5,
    )

    # save_subset(
    #     "monology/pile-uncopyrighted",
    #     split="train",
    #     filter_name="pile_set_name",
    #     filter_value="GitHub",
    #     max_bytes=50_000,
    #     output_file=os.path.join(OUTPUT_DIR, "github.jsonl"),
    #     max_samples=5,
    # )

    save_subset(
        "monology/pile-uncopyrighted",
        split="train",
        filter_name="pile_set_name",
        filter_value="Wikipedia (en)",
        max_bytes=20_000,
        output_file=os.path.join(OUTPUT_DIR, "wikipedia.jsonl"),
        max_samples=5,
    )

    save_subset(
        "allenai/c4",
        config="en",
        split="train",
        max_bytes=50_000,
        output_file=os.path.join(OUTPUT_DIR, "c4_en.jsonl"),
        max_samples=5,
    )
