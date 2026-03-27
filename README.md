# NK Military Activity Risk Analyst

Fine-tunes Qwen2.5-1.5B-Instruct with QLoRA to turn raw GDELT event data about North Korean military activity into structured JSON risk assessments. A knowledge-distillation approach uses Claude to generate training labels, then teaches the smaller model to replicate that behavior.

## Prerequisites

- Python 3.13
- [UV](https://docs.astral.sh/uv/) package manager
- An Anthropic API key (for label generation in step 3)
- A Google Cloud account **or** ~6 hours of patience (for the GDELT CSV fallback)
- NVIDIA GPU with >= 8 GB VRAM (training/inference). CPU works for steps 1-4.

## Setup

```bash
git clone <this-repo-url>
cd GenAI-hackathon

# Install dependencies
uv sync

# Create .env with your API keys
cp .env.example .env
# Then edit .env and fill in:
#   ANTHROPIC_API_KEY=sk-ant-...
#   HF_TOKEN=hf_...            (optional, for gated models on HuggingFace)
```

### BigQuery setup (recommended for step 1)

The fastest way to pull GDELT data is through Google BigQuery. The free tier (1 TB/month) is more than enough.

1. Create a Google Cloud project (free): https://console.cloud.google.com
2. Enable the BigQuery API
3. Authenticate locally:
   ```bash
   gcloud auth application-default login
   ```
4. Update the project ID in `01_fetch_gdelt.py` line 60 (`bigquery.Client(project=...)`) to your own GCP project ID.

If BigQuery isn't set up, the script automatically falls back to downloading daily CSV files directly from `data.gdeltproject.org`. This requires no authentication but takes several hours because it downloads one file per day across ~11 years.

## Reproducing the data pipeline

Run the numbered scripts in order. Each reads from the previous step's output.

```bash
# Step 1: Fetch NK military events from GDELT (BigQuery or CSV fallback)
uv run python 01_fetch_gdelt.py
# -> data/raw/nk_military_events.csv

# Step 2: Group events into weekly clusters
uv run python 02_build_clusters.py
# -> data/clusters/all_clusters.jsonl

# Step 3: Generate risk assessment labels via Claude API
# Requires ANTHROPIC_API_KEY in .env. Supports resuming if interrupted.
uv run python 03_generate_labels.py
# -> data/labeled/all_labeled.jsonl

# Step 4: Format into train/val/test splits
uv run python 04_prepare_dataset.py
# -> data/training/{train,val,test}.jsonl

# Step 5: QLoRA fine-tune
uv run python 05_finetune.py
# -> models/nk-risk-analyst/

# Step 6: Evaluate (before/after comparison on test set)
uv run python 06_evaluate.py
# -> data/evaluation_results.json
```

### Exact data reproduction notes

- **GDELT query parameters** are pinned in `config.py`:
  - Actor code: `PRK` (North Korea)
  - CAMEO root codes: `14`-`20` (military/conflict events)
  - Start date: `20150218` (GDELT v2 inception)
  - End date: the day you run the script
- **BigQuery vs CSV fallback may produce slightly different results**. BigQuery queries GDELT v2; the CSV fallback downloads GDELT v1 daily exports. Event IDs and counts can differ between versions.
- **Label generation (step 3) is non-deterministic** because it calls the Claude API with `temperature=0.3`. If you need the exact same labels, skip step 3 and use the existing `data/labeled/all_labeled.jsonl`. The script also supports resuming -- if interrupted, re-running it skips already-labeled clusters.
- **Dataset split (step 4) uses a fixed random seed**, so the train/val/test split is reproducible given the same input.

## Running the demo

```bash
uv run python app.py
```

Opens a Gradio web app with side-by-side comparison of the base model vs. the fine-tuned model. Includes pre-computed examples for instant display without GPU.

## Project structure

```
01_fetch_gdelt.py        # GDELT data retrieval (BigQuery or CSV)
02_build_clusters.py     # Weekly event clustering
03_generate_labels.py    # Claude API label generation
04_prepare_dataset.py    # Chat-template formatting and train/val/test split
05_finetune.py           # QLoRA fine-tuning
06_evaluate.py           # Before/after evaluation
app.py                   # Gradio demo app
config.py                # All configuration (model, data, training params)
DESIGN.md                # Detailed design document and rationale
data/                    # Generated data (not checked into git)
  raw/                   #   Raw GDELT events CSV
  clusters/              #   Weekly cluster JSONL
  labeled/               #   Claude-labeled risk assessments JSONL
  training/              #   Final train/val/test splits
models/                  # Fine-tuned model weights (not checked into git)
```

## Configuration

All tunable parameters live in `config.py`: model choice, LoRA hyperparameters, training settings, GDELT query filters, and dataset split ratios. See `DESIGN.md` for the reasoning behind each decision.
