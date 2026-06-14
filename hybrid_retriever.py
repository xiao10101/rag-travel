import os
import pickle
import jieba
from rank_bm25 import BM25Okapi
from milvus_manager import MilvusManager

BM25_INDEX_FILE = "bm25_index.pkl"
RRF_K = 60  # RRF 融合常数


def _tokenize(text: str) -> list[str]:
    """中文分词 + 英文空格分词混合"""
    # jieba 分词对中文和英文混合效果较好
    return list(jieba.cut(text))


class HybridRetriever:
    """BM25 + 向量混合检索，使用 RRF 融合排序"""

    def __init__(self):
        self.milvus = MilvusManager()
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []  # [{"text": ..., "page_num": ...}, ...]

    # 索引构建
    def build_index(self, chunks: list[dict]):
        """构建 BM25 索引并持久化到磁盘

        Args:
            chunks: [{"text": str, "page_num": int}, ...]
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        tokenized_corpus = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._save()
        print(f"BM25 索引构建完成，共 {len(chunks)} 个 chunk")

    def index_exists(self) -> bool:
        return os.path.exists(BM25_INDEX_FILE)

    def _save(self):
        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
            }, f)

    def _load(self) -> bool:
        if not os.path.exists(BM25_INDEX_FILE):
            return False
        with open(BM25_INDEX_FILE, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            texts = [c["text"] for c in self.chunks]
            tokenized_corpus = [_tokenize(t) for t in texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            return True

    # 混合检索
    def search(self, question: str, limit: int = 3) -> list[dict]:
        """执行混合检索，返回 RRF 融合排序后的结果

        Returns:
            [{"text": str, "page_num": int}, ...]
        """
        # 1. 向量检索（多取一些用于融合）
        vector_results = self.milvus.search(question, limit=limit * 3)

        # 2. BM25 检索
        bm25_results = self._bm25_search(question, limit=limit * 3)

        # 3. RRF 融合
        fused = self._rrf_merge(vector_results, bm25_results, limit)

        # 移除 distance 等内部字段，只返回外部需要的内容
        return [
            {"text": item["text"], "page_num": item["page_num"]}
            for item in fused
        ]

    # BM25 检索
    def _bm25_search(self, question: str, limit: int) -> list[dict]:
        if self.bm25 is None:
            if not self._load():
                print("BM25 索引未找到，仅使用向量检索")
                return []

        tokenized_query = _tokenize(question)
        scores = self.bm25.get_scores(tokenized_query)

        # 按 BM25 分数降序排列
        scored_indices = [
            (i, scores[i])
            for i in range(len(scores))
            if scores[i] > 0
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_indices[:limit]:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "page_num": chunk["page_num"],
                "_bm25_score": score,
            })

        return results

    @staticmethod
    def _rrf_merge(
        vector_results: list[dict],
        bm25_results: list[dict],
        limit: int,
    ) -> list[dict]:
        """Reciprocal Rank Fusion 融合排序"""
        # 建立 text → 融合分数 的映射
        rrf_scores: dict[str, dict] = {}

        # 为每个结果按 rank 计算 RRF 分数
        for rank, item in enumerate(vector_results):
            text = item["text"]
            rrf_scores.setdefault(text, {
                "text": text,
                "page_num": item["page_num"],
                "score": 0.0,
            })
            rrf_scores[text]["score"] += 1.0 / (RRF_K + rank + 1)

        for rank, item in enumerate(bm25_results):
            text = item["text"]
            rrf_scores.setdefault(text, {
                "text": text,
                "page_num": item["page_num"],
                "score": 0.0,
            })
            rrf_scores[text]["score"] += 1.0 / (RRF_K + rank + 1)

        # 按 RRF 分数降序排列
        sorted_items = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        return sorted_items[:limit]