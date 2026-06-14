from http import HTTPStatus

from dashscope import Generation


class LLMService:

    @staticmethod
    def rewrite_query(question: str) -> str:
        """将用户问题改写成更适合向量检索的查询"""
        prompt = f"""
你是一个查询改写助手。你的任务是将用户的问题改写成更适合搜索引擎和向量检索的形式。

规则：
1. 提取核心实体和关键概念
2. 移除口语化、冗余的表达
3. 保持原有语言
4. 输出简洁、信息密集的搜索查询

原始问题：
{question}

改写后的查询：
"""
        response = Generation.call(
            model="qwen-plus",
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
            result_format="message"
        )

        if response.status_code != HTTPStatus.OK:
            print("Query rewrite 调用失败，使用原始问题")
            return question

        rewritten = response.output.choices[0].message.content.strip()
        print(f"查询改写: {question} → {rewritten}")
        return rewritten

    @staticmethod
    def chat(question: str, contexts: list[dict]):
        """生成回答，contexts 为 [{"text": ..., "page_num": ...}, ...]"""
        # 格式化上下文，为每段标注来源页码
        context_lines = []
        for i, ctx in enumerate(contexts, start=1):
            context_lines.append(
                f"[片段 {i} (第 {ctx['page_num']} 页)]\n{ctx['text']}"
            )
        formatted_context = "\n\n".join(context_lines)

        prompt = f"""
你是一个基于文档的问答助手。

请严格根据提供的上下文回答问题。回答时必须遵守以下格式要求：

1. 引用来源：当使用某个片段中的信息时，在句末标注 [页码 X]
2. 多源引用：如果多个片段都支持同一结论，列出所有页码 [页码 3][页码 5]
3. 无法回答：如果上下文里没有答案，请明确说"我无法从提供的文档中找到答案"

================
上下文（每段标注了来源页码）：
{formatted_context}
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