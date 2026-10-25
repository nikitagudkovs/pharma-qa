import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CACHE_DIR = Path(os.getenv("CACHE_DIR", DATA_DIR / "cache"))
TRACE_DIR = Path(os.getenv("TRACE_DIR", ROOT / "traces"))
LABELS_PATH = DATA_DIR / "labels.jsonl"
SYNONYMS_PATH = DATA_DIR / "synonyms.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "claude-haiku-5-5")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "claude-sonnet-5-5")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.getenv("USE_RERANKER", "0") == "1"

OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY") or None
# Per-IP limit on POST /ask: requests per window
ASK_RATE_LIMIT = int(os.getenv("ASK_RATE_LIMIT", "10"))
ASK_RATE_WINDOW_S = int(os.getenv("ASK_RATE_WINDOW_S", "60"))
