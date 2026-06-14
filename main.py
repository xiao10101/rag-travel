
from llm import LLMService
from milvus_manager import MilvusManager


def main():
    manager = MilvusManager()
    question = '公司的年假是怎么计算的？假如我入职4个月，我能有几天年假？'

    # Step 1: Rewrite — 改写查询以提高检索质量
    rewritten_query = LLMService.rewrite_query(question)

    # Step 2: Retrieve — 用改写后的查询检索（返回带页码的上下文）
    contexts = manager.search(rewritten_query)

    # Step 3: Read — 传入结构化上下文，LLM 会自动标注引用
    answer = LLMService.chat(
        question=question,
        contexts=contexts
    )

    print("回答:")
    print(answer)


if __name__ == "__main__":
    main()