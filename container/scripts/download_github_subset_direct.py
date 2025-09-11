import gzip
import json
import os

from huggingface_hub import hf_hub_download


def download_github_subset(
    repo_id="monology/pile-uncopyrighted",
    filename="data/github.jsonl.zst",  # GitHub shard inside the repo
    max_bytes=0,
    output_file="github.jsonl",
    max_samples=None,
):
    """
    Download GitHub subset directly from The Pile dataset repo.
    Much faster than streaming + filtering.
    """

    # Download file from HF Hub (cached locally after first time)
    print("⬇️ Downloading GitHub shard...")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    total_bytes, collected = 0, 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        # Decompress if file is zst/gz
        if local_path.endswith(".gz"):
            opener = gzip.open(local_path, "rt", encoding="utf-8")
        else:
            import zstandard as zstd

            opener = zstd.open(open(local_path, "rb"), "rt", encoding="utf-8")

        with opener as f:
            for line in f:
                if max_samples and collected >= max_samples:
                    break

                encoded = line.encode("utf-8")
                if max_bytes and (total_bytes + len(encoded) > max_bytes):
                    break

                out_f.write(line)
                total_bytes += len(encoded)
                collected += 1

    print(f"✅ Saved {collected} GitHub samples ({total_bytes} bytes) to {output_file}")


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    download_github_subset(
        max_bytes=50_000,
        output_file=os.path.join(OUTPUT_DIR, "github.jsonl"),
        max_samples=5,
    )
