"""Central configuration for the NK Military Activity Risk Analyst project."""

from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLUSTERS_DIR = DATA_DIR / "clusters"
LABELED_DIR = DATA_DIR / "labeled"
TRAINING_DIR = DATA_DIR / "training"
MODELS_DIR = PROJECT_ROOT / "models" / "nk-risk-analyst"

# ── GDELT configuration ───────────────────────────────────────────────────
GDELT_BQ_TABLE = "gdelt-bq.gdeltv2.events"
GDELT_START_DATE = 20150218  # GDELT v2 starts Feb 18, 2015

# North Korea CAMEO actor code (PRK = DPRK, NOT KOR which is South Korea)
NK_ACTOR_CODE = "PRK"

# CAMEO root codes for military/conflict events
CAMEO_MILITARY_ROOT_CODES = ["14", "15", "16", "17", "18", "19", "20"]

# Expanded codes if we need more data (diplomatic conflict)
CAMEO_EXPANDED_ROOT_CODES = ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

# Fields to extract from GDELT
GDELT_FIELDS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "Actor1CountryCode",
    "Actor1Name",
    "Actor1Type1Code",
    "Actor2CountryCode",
    "Actor2Name",
    "Actor2Type1Code",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_FullName",
    "Actor2Geo_FullName",
    "ActionGeo_FullName",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "SOURCEURL",
]

# CAMEO root code descriptions (for readable cluster summaries)
CAMEO_DESCRIPTIONS = {
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit Force",
    "16": "Reduce Military",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Unconventional Violence",
}

# ── Clustering ─────────────────────────────────────────────────────────────
MIN_EVENTS_PER_CLUSTER = 3  # Drop weeks with fewer events than this
CLUSTER_FREQ = "W"  # Weekly windows

# ── Model configuration ───────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# QLoRA / bitsandbytes
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
MAX_SEQ_LENGTH = 1024
MAX_GRAD_NORM = 0.3

# ── Label generation ──────────────────────────────────────────────────────
LABEL_MODEL = "claude-sonnet-4-20250514"
LABEL_TEMPERATURE = 0.3

# Risk assessment output schema fields
RISK_ASSESSMENT_FIELDS = [
    "escalation_level",
    "escalation_rationale",
    "situation_summary",
    "key_actors",
    "historical_context",
    "watch_indicators",
    "potential_trajectories",
    "confidence_level",
    "data_caveats",
]

# ── Dataset split ─────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
