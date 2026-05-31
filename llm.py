from http import HTTPStatus

from dashscope import Generation

class LLMService:

    @staticmethod
    def chat(question: str, context: str):
        prompt = f"""
你是一个基于文档的问答助手。

请严格根据提供的上下文回答问题。

如果上下文里没有答案，请明确说：
“我无法从提供的文档中找到答案”。

================
上下文：
{context}
================

问题：
{question}
"""
        response = Generation.call(
            model="qwen-plus",
            temperature=0.2,
            top_p=0.8,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的RAG问答助手"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            result_format="message"
        )

        if response.status_code != HTTPStatus.OK:
            print("LLM 调用失败")
            print("status_code:", response.status_code)
            print("code:", response.code)
            print("message:", response.message)

            raise Exception(response.message)
        return (
            response
            .output
            .choices[0]
            .message
            .content
        )