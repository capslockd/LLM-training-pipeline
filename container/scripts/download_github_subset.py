import json
import os

from datasets import load_dataset


def save_github_subset(
    output_file="./data/github_50mb.jsonl",
    max_bytes=50_000_000,  # 50 MB
    buffer_size_lines=1000,  # buffer lines before writing
):
    """
    Efficiently download GitHub subset from The Pile (~50MB) using streaming, shuffle, and buffered writes.
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Streaming load + shuffle to bring GitHub samples earlier
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=100_000, seed=42)  # adjust buffer for speed/memory

    total_bytes = 0
    collected = 0
    buffer_lines = []

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in ds:
            meta = sample.get("meta", {})
            if meta.get("pile_set_name") != "GitHub":
                continue  # skip non-GitHub samples

            text = sample.get("text", "")
            encoded = text.encode("utf-8")

            if max_bytes and (total_bytes + len(encoded)) > max_bytes:
                break

            buffer_lines.append(json.dumps({"text": text}))
            total_bytes += len(encoded)
            collected += 1

            # write buffer to file
            if len(buffer_lines) >= buffer_size_lines:
                f.write("\n".join(buffer_lines) + "\n")
                buffer_lines = []

        # write any remaining lines
        if buffer_lines:
            f.write("\n".join(buffer_lines) + "\n")

    print(
        f"✅ Saved {collected} GitHub samples ({total_bytes / 1e6:.2f} MB) to {output_file}"
    )


if __name__ == "__main__":
    save_github_subset()
