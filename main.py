
from llm import LLMService
from hybrid_retriever import HybridRetriever


def main():
    retriever = HybridRetriever()
    question = '公司的年假是怎么计算的？假如我入职4个月，我能有几天年假？'

    # Step 1: Rewrite — 改写查询以提高检索质量
    rewritten_query = LLMService.rewrite_query(question)

    # Step 2: Retrieve — 混合检索（BM25 + 向量 + RRF 融合）
    contexts = retriever.search(rewritten_query)

    # Step 3: Read — 传入结构化上下文，LLM 会自动标注引用
    answer = LLMService.chat(
        question=question,
        contexts=contexts
    )

    print("回答:")
    print(answer)


if __name__ == "__main__":
    main()