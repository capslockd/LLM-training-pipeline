# Mainpipe Dataset Pipeline — Design Report

## 1. Overview
The **Mainpipe pipeline** is an end-to-end data preparation system designed to:

1. **Retrieve datasets** (PubMed Abstracts, Wikipedia, GitHub code, C4 web text).  
2. **Preprocess** text (cleaning, normalization, tokenization).  
3. **Mix datasets** according to configurable ratios.  
4. **Shard datasets** into training-ready JSONL or Arrow files.  
5. **Enable inspectability and debugging** through metrics and sample reports.  
6. **Run fully inside Docker** for reproducibility.  
7. **Be configurable via YAML** for easy environment adjustments.

---

## 2. Architecture

### Core Components

- **`RetrieveDatasets` (`utils/download_dataset.py`)**  
  - Handles dataset download and filtering.  
  - `_save_subset`: streams from HuggingFace Hub, filters, size-limits, saves as `.jsonl`.  
  - `download_datasets`: orchestrates downloads for PubMed, Wikipedia, C4, and optionally GitHub.  

- **`DataParser` (`utils/data_parser.py`)**  
  - Handles **text-level processing**:  
    - `clean_text`: removes noise, lowercases, removes unwanted characters.  
    - `normalize_text`: deduplicates punctuation, normalizes numbers.  
    - `tokenize_text`: converts text into `input_ids` using HuggingFace tokenizer.  
  - Default tokenizer: `bert-base-uncased` (configurable).

- **`DatasetPreper` (`utils/data_prep.py`)**  
  - Handles **dataset-level processing**:  
    - Load processed JSONL files.  
    - Mix datasets with ratios.  
    - Shard into `.jsonl` or `.arrow`.  
    - Run **debug sampling** or **inspectability checks** (histograms, PII detection, duplicates).

- **`main.py`**  
  - Orchestration layer:  
    - Downloads datasets via `RetrieveDatasets`.  
    - Cleans/normalizes/tokenizes via `DataParser`.  
    - Loads/mixes/shards via `DatasetPreper`.  
    - Produces debug and inspect outputs if enabled.  

---
*** Pipeline Flow Diagram ***
```mermaid
flowchart TD
    A[Start: Docker / CLI Run] --> B[RetrieveDatasets.download_datasets()]
    B --> C[DataParser: Clean & Normalize Text]
    C --> D[DataParser: Tokenize Text (input_ids)]
    D --> E[DatasetPreper: Load Processed JSONL Files]
    E --> F[DatasetPreper: Mix Datasets with Configurable Ratios]
    F --> G[DatasetPreper: Shard into JSONL / Arrow Files]
    G --> H{Inspect & Debug?}
    H -- Yes --> I[Generate Inspect Reports (Length, PII, Duplicates, Language)]
    H -- No --> J[Skip Inspect]
    I --> K[Output: Shards + Inspect Reports]
    J --> K[Output: Shards Only]
    K --> L[End: Ready for Model Training]
```

## 3. Configuration

All pipeline parameters are centralized in `config/mainpipe_nonprod.yaml`.  

```yaml
datasets:
  - pubmed.jsonl
  - wikipedia.jsonl
  - c4_en.jsonl
  # - github.jsonl

preprocessing:
  tokenizer: "bert-base-uncased"
  max_length: 512
  normalize_numbers: true
  lowercase: true

mixing:
  ratios:
    pubmed: 0.25
    wikipedia: 0.25
    c4: 0.25
  shuffle_seed: 42

sharding:
  shard_size: 1000
  as_arrow: false
  prefix: "shard"

# Debugging (fast, lightweight)
debug:
  enabled: true
  inspect_samples: 5
  verbose: true

# Inspectability (full dataset-level checks)
inspect:
  enabled: true
  output_dir: "inspect_reports"
  detect_language: true
  check_duplicates: true
  check_pii: true

```
---

## 4. How to run (Docker + CLI)

**Build image (no downloads at build-time)**
Make sure Dockerfile does not run the heavy pipeline during build.

```bash
# build image
docker compose build

```

Run pipeline (recommended — runs inside container, writes to host via volumes)

Mount host folders so outputs persist on host:

```bash
docker run --rm \        
  -v $(pwd)/data:/usr/scripts/data \
  -v $(pwd)/processed:/usr/scripts/processed \
  -v $(pwd)/processed_shards:/usr/scripts/processed_shards \
  -v $(pwd)/inspect_reports:/usr/scripts/inspect_reports \
  container-mainpipe \
  python main.py
```
---

### 5. Outputs (files & folders)

	-	data/ — raw downloaded JSONL files:
	    - pubmed.jsonl (~50MB target)
	    - wikipedia.jsonl (~20MB target)
	    - c4_en.jsonl (~50MB target)
	    - optionally github.jsonl (configurable / max_samples recommended)

    - processed/ — intermediate *_processed.jsonl containing:
	    - "text" (cleaned + normalized)
	    - "input_ids" (token IDs)

    - processed_shards/ — final training shards:
	    - shard_000.jsonl, shard_001.jsonl, …
	    - optionally Arrow folders shard_000_arrow/ if as_arrow: true
    
    - inspect_reports/ — inspectability outputs:
	    - inspect_report.json (summary statistics)
	    - length_hist.png (histogram)
	    - other artifacts (optionally per-shard stats)
---

### 6. Testing, Validation & Inspectability

*** Debug / Inspect ***

	-	Debug (fast): sample previews (first n processed samples). Controlled by debug.enabled and inspect_samples.
	-	Inspect (deeper): corpus-level metrics such as:
	-	Token/char length histogram.
	-	Language distribution (sampled, langdetect).
	-	Duplicate detection via hashing.
	-	PII hit counts (email/phone regex heuristics).

---

### Unfinished tasks

	-   GitHub subset: streaming+filtering The Pile is slow. Options:
	    - Maybe could use a pre-split code dataset (e.g., bigcode/the-stack, codeparrot/github-code).
	    
	-   Tokenizer choice: bert-base-uncased is a general-purpose tokenizer (good baseline). For domain-specific tasks (biomedical, code) use specialized tokenizers (PubMedBERT, CodeBERT) or other model specific tokenizers.