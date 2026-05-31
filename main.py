
from llm import LLMService
from milvus_manager import MilvusManager

def main():
    manager = MilvusManager()
    question = "内容管理后台的网址是什么"
    contexts = manager.search(question)

    context_text = "\n".join(contexts)

    answer = LLMService.chat(
        question=question,
        context=context_text
    )

    print("搜索结果:")
    print(answer)

if __name__ == '__main__':
    main()