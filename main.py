
from llm import LLMService
from milvus_manager import MilvusManager

def main():
    manager = MilvusManager()
    question = "公司可以“即时辞退”员工且不支付经济补偿的，列举其中5种情形。"
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