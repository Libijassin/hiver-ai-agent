"""
Central configuration for the Hiver AI Customer Support Agent.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Brand Selection ──────────────────────────────────────────────────────────
BRAND_NAME = "SpotifyCares"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "twcs" / "twcs.csv"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Settings ───────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.8-flash"
USE_MOCK = not bool(GEMINI_API_KEY)  # Auto-detect: use mock if no API key

# ── Retrieval Settings ───────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 5
RETRIEVAL_MIN_SIMILARITY = 0.15  # Minimum cosine similarity to consider

# ── Escalation Thresholds ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5       # Below this → escalate
RETRIEVAL_SIM_THRESHOLD = 0.1    # Below this → escalate (insufficient evidence)

# ── Data Split Settings ──────────────────────────────────────────────────────
EVAL_SPLIT_RATIO = 0.2           # 20% for evaluation
RANDOM_SEED = 42

# ── Golden Set Settings ──────────────────────────────────────────────────────
GOLDEN_SET_SIZE = 200            # Target size for golden evaluation set
