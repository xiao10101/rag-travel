import os

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

MILVUS_URI = "http://localhost:19530"

COLLECTION_NAME = "demo_collection"

EMBEDDING_MODEL = "text-embedding-v2"

RERANK_MODEL = "qwen3-rerank"
RERANK_TOP_N = 5

# ---- 项目路径 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BM25_INDEX_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")
GOLDEN_DATA_FILE = os.path.join(DATA_DIR, "golden_data.json")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")