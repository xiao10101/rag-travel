from http import HTTPStatus

from dashscope import TextReRank
import dashscope

from app.config import DASHSCOPE_API_KEY, RERANK_MODEL

dashscope.api_key = DASHSCOPE_API_KEY


class Reranker:
    """基于 DashScope qwen3-rerank 模型的精排器"""

    @staticmethod
    def rerank(query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
        """对候选文档进行精排

        Args:
            query: 用户查询
            documents: 候选文档列表 [{"text": str, "page_num": int, ...}, ...]
            top_n: 返回前 N 个最相关的文档

        Returns:
            精排后的文档列表 [{"text": str, "page_num": int, "rerank_score": float}, ...]
        """
        if not documents:
            return []

        doc_texts = [doc["text"] for doc in documents]

        resp = TextReRank.call(
            model=RERANK_MODEL,
            query=query,
            documents=doc_texts,
            top_n=top_n,
            return_documents=False,
        )

        if resp.status_code != HTTPStatus.OK:
            print(f"ReRank 调用失败: {resp.message}")
            # 降级：返回原始顺序
            return documents[:top_n]

        reranked = []
        for result in resp.output.results:
            original_doc = documents[result.index]
            reranked.append({
                **original_doc,
                "rerank_score": result.relevance_score,
            })

        return reranked