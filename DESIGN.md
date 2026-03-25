# Design Document: NK Military Activity Risk Analyst

## Project Overview

Fine-tune Llama-3.2-3B-Instruct with QLoRA to transform raw GDELT event data about North Korean military activity into structured geopolitical risk assessments. The model learns to behave like an intelligence analyst: given a week's worth of coded news events, it produces a JSON risk assessment with escalation level, situation summary, key actors, and watch indicators.

---

## Data Decisions

### Why GDELT?

GDELT (Global Database of Events, Language, and Tone) is the world's largest open event database. It processes news from 65+ languages every 15 minutes and codes each event using the CAMEO (Conflict and Mediation Event Observations) taxonomy.

**What we get from GDELT**:
- Structured event records (not raw text) — each row is "Actor A did Action X to Actor B at Location Y"
- Quantitative scores: Goldstein Scale (-10 to +10, conflict to cooperation), tone, mention counts
- Geographic coordinates for every event
- Source URLs back to original articles
- 15+ years of historical data

**Why not satellite imagery?** Commercial providers (Maxar) cost thousands. Free alternatives (Sentinel-2 at 10m/pixel) can't resolve individual vehicles. Object detection on satellite imagery is a computer vision problem, not an LLM fine-tuning task — doesn't meet the hackathon rubric.

**Why not GDELT's Visual GKG (image data)?** The images indexed by VGKG are mostly article thumbnails, stock photos, and low-quality news imagery. Classifying these as "military convoys" would produce noise, not signal.

### GDELT Data Access

**Primary method: Google BigQuery**
- GDELT maintains a public BigQuery dataset: `gdelt-bq.gdeltv2.events`
- Free tier allows 1 TB/month of queries (our query is ~10 GB)
- Fastest and most reliable access method
- Requires a Google Cloud account (free)

**Fallback method: Direct CSV download**
- GDELT publishes raw CSV files at `data.gdeltproject.org`
- Files are tab-separated, headerless, 61 columns
- Column positions defined by the GDELT 2.0 Event Database Codebook
- Slower but requires no authentication

### Actor Code: PRK vs KOR

This is a critical distinction that was wrong in the original Copilot plan:
- **`PRK`** = Democratic People's Republic of Korea (North Korea) — this is what we want
- **`KOR`** = Republic of Korea (South Korea) — this is NOT North Korea

CAMEO country codes follow ISO 3166-1 alpha-3 conventions. We filter:
- `Actor1CountryCode = 'PRK'` → North Korea as the initiator of an action
- `Actor2CountryCode = 'PRK'` → North Korea as the target of an action
- Both directions capture the full picture of NK military activity

### CAMEO Event Codes

The CAMEO taxonomy organizes events into a hierarchy. We focus on military/conflict codes:

| Root Code | Category | Examples |
|-----------|----------|----------|
| 14 | Protest | Mass protests, demonstrations |
| 15 | Exhibit Force | Military exercises, shows of force, mobilizations |
| 16 | Reduce Military | Troop withdrawals, demobilization |
| 17 | Coerce | Threats, sanctions, blockades |
| 18 | Assault | Targeted attacks, bombings |
| 19 | Fight | Armed clashes, conventional warfare |
| 20 | Unconventional Mass Violence | WMD use, ethnic cleansing |

Each root code has sub-codes (e.g., 152 = "Mobilize or increase military forces", 190 = "Use conventional military force"). We capture all sub-codes.

**Fallback expansion**: If data is sparse, we can include codes 10-13 (Demand, Disapprove, Reject, Threaten) which capture diplomatic conflict escalation.

### Key GDELT Fields

| Field | Type | Purpose |
|-------|------|---------|
| GLOBALEVENTID | int | Unique event identifier |
| SQLDATE | int (YYYYMMDD) | Event date |
| Actor1CountryCode | str | Country code of Actor 1 |
| Actor1Name | str | Name of Actor 1 (e.g., "NORTH KOREA", "KIM JONG UN") |
| Actor2CountryCode | str | Country code of Actor 2 |
| Actor2Name | str | Name of Actor 2 |
| EventCode | str | Full CAMEO event code (e.g., "152") |
| EventRootCode | str | Root CAMEO code (e.g., "15") |
| GoldsteinScale | float | Conflict/cooperation score (-10 to +10) |
| AvgTone | float | Average tone of source articles (-100 to +100) |
| NumMentions | int | Number of article mentions |
| NumSources | int | Number of distinct sources |
| ActionGeo_FullName | str | Location name |
| ActionGeo_Lat/Long | float | Geographic coordinates |
| SOURCEURL | str | URL of first source article |

### Date Range Decision

**15 years** (~2011-2026) instead of the original 4 years, because:
- More data = more weekly clusters = better fine-tuning (500-600 clusters vs 150-200)
- Captures major NK escalation cycles: 2013 nuclear test, 2017 ICBM tests, 2022-23 missile surge
- Historical diversity teaches the model to distinguish routine posturing from genuine escalation
- GDELT v1 covers 2013+, GDELT v2 covers Feb 2015+ (we'll use v2 primarily, ~11 years)

**Expected data volume**: ~15,000-40,000 individual events, yielding ~500-600 non-empty weekly clusters.

### Weekly Clustering Rationale

Events are grouped into 7-day windows because:
- Individual events are too granular (a single news mention) for meaningful risk assessment
- Monthly windows are too coarse (miss rapid escalation sequences)
- Weekly aligns with how intelligence briefings are typically structured
- Produces enough clusters (500+) for a solid training set

Weeks with fewer than 3 events are dropped (too sparse for meaningful assessment).

---

## Model Architecture Decisions

### Why Llama-3.2-3B-Instruct?

| Factor | Llama-3.2-3B | Mistral-7B | Phi-3-mini (3.8B) |
|--------|-------------|------------|-------------------|
| Parameters | 3B | 7B | 3.8B |
| VRAM (4-bit) | ~2.5 GB | ~5 GB | ~3 GB |
| Training time | ~15 min | ~30 min | ~20 min |
| Headroom on 8GB GPU | 5.5 GB free | 3 GB free | 5 GB free |
| Instruct-tuned | Yes | Yes | Yes |
| License | Llama 3.2 Community | Apache 2.0 | MIT |

**Decision**: Llama-3.2-3B-Instruct. It's Meta's latest small model, already instruction-tuned (so it understands prompts well), fits comfortably on our RTX 4060 (8 GB) with plenty of headroom for batch processing, and trains fast enough for iterative hackathon development.

### Why QLoRA (Quantized Low-Rank Adaptation)?

**The problem**: Full fine-tuning updates all 3 billion parameters. This requires:
- Storing model weights (~6 GB in fp16)
- Storing optimizer states (~12 GB for Adam)
- Storing gradients (~6 GB)
- Total: ~24 GB — 3x our GPU memory

**LoRA solution**: Freeze the original weights. For each target weight matrix W, add two small matrices A and B such that the effective weight becomes W + BA. If W is 4096×4096 but A is 4096×16 and B is 16×4096, we only train ~130K parameters instead of 16M per layer. Across all target modules, LoRA adds ~1-2% new trainable parameters.

**Quantization on top (the "Q" in QLoRA)**: Store the frozen original weights in 4-bit precision instead of 16-bit. This cuts the frozen model's memory footprint by 4x:
- 3B params in fp16 = ~6 GB
- 3B params in 4-bit = ~1.5 GB

**Combined effect**:
- Frozen model in 4-bit: ~1.5 GB
- LoRA adapter parameters: ~50 MB
- Optimizer states for LoRA only: ~100 MB
- Gradients: ~50 MB
- **Total: ~2.5 GB** — fits easily in 8 GB with room for batch processing

### LoRA Configuration

```python
LoraConfig(
    r=16,              # Rank of the low-rank matrices (higher = more capacity, more memory)
    lora_alpha=32,     # Scaling factor (alpha/r = 2 is a common effective learning rate multiplier)
    lora_dropout=0.05, # Dropout on LoRA layers (light regularization)
    target_modules=[   # Which layers get LoRA adapters:
        "q_proj",      #   Query projection (attention)
        "k_proj",      #   Key projection (attention)
        "v_proj",      #   Value projection (attention)
        "o_proj",      #   Output projection (attention)
        "gate_proj",   #   Gate projection (MLP)
        "up_proj",     #   Up projection (MLP)
        "down_proj",   #   Down projection (MLP)
    ],
    task_type="CAUSAL_LM",
)
```

**Why these target modules?** We apply LoRA to both attention layers (how the model focuses on different parts of the input) and MLP/feed-forward layers (where the model processes and transforms information). Targeting all of these gives LoRA enough capacity to learn our domain-specific behavior without being so large that it overfits on ~400 training examples.

### Quantization Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,              # Use 4-bit quantization
    bnb_4bit_quant_type="nf4",      # Normal Float 4 — optimized for normally-distributed weights
    bnb_4bit_compute_dtype=bfloat16, # Compute in bfloat16 for stability
    bnb_4bit_use_double_quant=True,  # Quantize the quantization constants too (saves ~0.4 GB)
)
```

### Training Hyperparameters

```python
TrainingArguments(
    num_train_epochs=3,           # 3 passes through the training data
    per_device_train_batch_size=4, # 4 examples per GPU step
    gradient_accumulation_steps=2, # Effective batch size = 4×2 = 8
    learning_rate=2e-4,           # Standard for LoRA fine-tuning
    warmup_steps=10,              # Gradual learning rate warmup
    lr_scheduler_type="cosine",   # Cosine decay — smooth learning rate reduction
    bf16=True,                    # bfloat16 mixed precision
    max_grad_norm=0.3,            # Gradient clipping for stability
)
```

**Why 3 epochs?** With ~400 training examples, 3 epochs means the model sees each example 3 times. Too few and it won't learn the format; too many and it memorizes the training data (overfitting). 3 is the sweet spot for LoRA on small datasets.

**Why batch size 4 with grad accum 2?** Effective batch size of 8 balances training stability (larger batches = smoother gradients) with memory constraints. 4 examples at ~2048 tokens each fit comfortably in 8 GB VRAM.

---

## Training Data Pipeline

### Knowledge Distillation Approach

We don't have human-written risk assessments paired with GDELT data. Nobody has created this dataset. So we use **knowledge distillation**: a powerful model (Claude) generates the training labels, and we teach a smaller model (Llama-3.2-3B) to replicate that behavior.

**Pipeline**:
1. GDELT events → weekly clusters (structured text summaries)
2. Weekly clusters → Claude API → JSON risk assessments (our training labels)
3. (cluster, assessment) pairs → formatted into Llama chat template → fine-tuning dataset

### Training Data Format

Each training example follows the Llama-3.2 chat template:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a geopolitical risk analyst specializing in North Korean military activity. Analyze GDELT event data and produce structured risk assessments in JSON format.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Week: 2024-03-11 to 2024-03-17
Total Events: 47
Event Breakdown: Exhibit Force(23), Coerce(12), Assault(8), Fight(4)
Avg Goldstein Scale: -7.2
Avg Tone: -4.8
Key Actors: North Korea (KPA), United States (USFK), South Korea (ROK Army)
Locations: Pyongyang, Sea of Japan, Korean DMZ
...<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{
  "escalation_level": 3,
  "escalation_rationale": "...",
  "situation_summary": "...",
  "key_actors": [...],
  "watch_indicators": [...],
  "confidence_level": "medium"
}<|eot_id|>
```

### Risk Assessment Schema

The output JSON has these fields:
- **escalation_level** (1-5): 1=routine posturing, 2=elevated rhetoric, 3=provocative actions, 4=significant military escalation, 5=active conflict
- **escalation_rationale**: 2-3 sentences explaining the rating
- **situation_summary**: 3-5 sentence assessment of the week's developments
- **key_actors**: List of {name, role, posture} objects
- **historical_context**: 1-2 sentences connecting to broader patterns
- **watch_indicators**: 3-5 specific things to monitor going forward
- **potential_trajectories**: 2-3 possible near-term scenarios
- **confidence_level**: "low", "medium", or "high" based on data quality
- **data_caveats**: 1-2 sentences on limitations of this assessment

### Dataset Split

- **Train**: 80% (~400 examples) — model learns from these
- **Validation**: 10% (~50 examples) — monitors overfitting during training
- **Test**: 10% (~50 examples) — final before/after evaluation, never seen during training

---

## Hardware Constraints

### RTX 4060 Laptop GPU — 8 GB VRAM

| Component | VRAM Usage |
|-----------|-----------|
| Llama-3.2-3B in 4-bit | ~1.5 GB |
| LoRA adapter weights | ~50 MB |
| Optimizer states (AdamW for LoRA params) | ~100 MB |
| Gradients | ~50 MB |
| Batch of 4 × 2048 tokens | ~500 MB |
| CUDA overhead | ~300 MB |
| **Total during training** | **~2.5 GB** |
| **Free headroom** | **~5.5 GB** |

This gives us comfortable margin. If we hit OOM for any reason:
- Reduce batch size to 2 (saves ~250 MB)
- Reduce max sequence length to 1024 (saves ~250 MB)
- Increase gradient accumulation to maintain effective batch size

### Training Time Estimate

- ~400 training examples × 3 epochs = 1,200 steps / effective batch 8 = 150 optimizer steps
- RTX 4060 processes ~2 steps/second with 3B model
- **Estimated training time: ~10-15 minutes**

---

## Evaluation Strategy

### Before/After Comparison

For each test example, we run inference with:
1. **Base Llama-3.2-3B-Instruct** (no LoRA adapter) — the "before"
2. **Fine-tuned model** (base + LoRA adapter merged) — the "after"

### Metrics

| Metric | What it measures | Expected result |
|--------|-----------------|-----------------|
| Format compliance | Does output parse as valid JSON with all required fields? | Base: ~10-20%, Fine-tuned: ~85-95% |
| Escalation accuracy | MAE between predicted and "ground truth" escalation level | Base: random (~2.0 MAE), Fine-tuned: ~0.5-1.0 MAE |
| Field completeness | Average % of required fields present and non-empty | Base: ~30%, Fine-tuned: ~90%+ |
| Qualitative | Side-by-side reading of 5 examples | Base: vague/generic, Fine-tuned: specific/structured |

---

## Demo Application

Gradio web app with:
- **Input**: Dropdown to select a test cluster, or paste custom GDELT event summary
- **Two output panels**: "Base Model (Before)" and "Fine-tuned Model (After)"
- **Generate button**: Runs both models on the same input
- **Pre-computed tab**: 5 cached examples for instant demo (no GPU wait during presentation)
- **Ethics tab**: Summary of risks and limitations

---

## Ethics and Limitations

### Hallucination Risk
The model may generate specific military details (unit names, weapon systems, troop counts) that are not present in the GDELT data. In a real intelligence context, this could lead to false threat assessments. **Mitigation**: confidence scores, source citation links, human-in-the-loop validation.

### Training Label Bias
Our "ground truth" labels were generated by Claude, not human intelligence analysts. The fine-tuned model inherits:
- Claude's analytical framework and biases
- Claude's interpretation of escalation levels
- Any systematic errors in Claude's geopolitical reasoning

### GDELT Source Bias
GDELT overrepresents English-language and Western news sources. Coverage of internal DPRK events is inherently limited — we see what foreign journalists report, not what actually happens inside North Korea. Events may reflect media attention cycles rather than actual military activity changes.

### Dual-Use Concerns
An automated geopolitical risk scoring tool could be misused for:
- Information warfare (generating convincing but misleading threat assessments)
- Algorithmic trading on geopolitical events (market manipulation)
- Justifying predetermined policy positions with AI-generated "analysis"
- Surveillance and monitoring of specific nations without appropriate oversight

### Calibration Limitation
The 1-5 escalation scale was defined and applied by an LLM, not validated against historical outcomes (e.g., did events rated "4" actually precede military confrontations?). This scale should not be treated as a validated early warning indicator comparable to established systems like ACLED's CAST or the Global Conflict Risk Index.
