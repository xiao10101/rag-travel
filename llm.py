from http import HTTPStatus
from typing import Generator

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
    def chat(question: str, contexts: list[dict]) -> str:
        """同步生成回答（非流式），contexts 为 [{"text": ..., "page_num": ...}, ...]"""
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

    @staticmethod
    def chat_stream(
        question: str,
        contexts: list[dict],
    ) -> Generator[dict, None, None]:
        """流式生成回答，带思考过程展示

        Yields:
            dict 包含:
            - type: "thinking" | "content" | "done" | "error"
            - content: str (当前 chunk 的内容)
            - error: str (仅当 type="error" 时)
        """
        # 格式化上下文
        context_lines = []
        for i, ctx in enumerate(contexts, start=1):
            context_lines.append(
                f"[片段 {i} (第 {ctx['page_num']} 页)]\n{ctx['text']}"
            )
        formatted_context = "\n\n".join(context_lines)

        system_prompt = """你是一个专业的RAG问答助手。在回答用户问题之前，请先按照以下步骤进行思考：

## 思考步骤（必须输出）：
1. **理解问题**：分析用户问题的核心意图和关键实体
2. **检索分析**：评估提供的上下文片段与问题的相关性
3. **证据提取**：从相关片段中提取关键信息和事实
4. **逻辑推理**：基于证据进行推理，构建回答框架
5. **答案组织**：组织最终回答，并标注引用来源

请严格按照以下格式输出你的回答：

<thinking>
【在此输出你的思考过程】
- 理解问题：...
- 检索分析：...
- 证据提取：...
- 逻辑推理：...
</thinking>

【最终回答】
（在此输出正式回答，引用时标注 [页码 X]）
"""

        user_prompt = f"""================
上下文（每段标注了来源页码）：
{formatted_context}
================

问题：
{question}"""

        responses = Generation.call(
            model="qwen-plus",
            temperature=0.2,
            top_p=0.8,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            result_format="message",
            stream=True,
            incremental_output=True,
        )

        # 状态机：解析 thinking 和 content
        in_thinking = False
        thinking_buffer = ""
        content_buffer = ""
        thinking_complete = False

        try:
            for response in responses:
                if response.status_code != HTTPStatus.OK:
                    yield {
                        "type": "error",
                        "content": "",
                        "error": f"API 错误: {response.message} (code: {response.code})",
                    }
                    return

                if not response.output.choices:
                    continue

                chunk = response.output.choices[0].message.content

                if not chunk:
                    continue

                # 解析 thinking 标签
                if "<thinking>" in chunk and not thinking_complete:
                    in_thinking = True
                    # 提取 <thinking> 之后的内容
                    parts = chunk.split("<thinking>", 1)
                    if len(parts) > 1:
                        chunk = parts[1]

                if "</thinking>" in chunk and in_thinking:
                    in_thinking = False
                    thinking_complete = True
                    parts = chunk.split("</thinking>", 1)
                    thinking_buffer += parts[0]
                    if len(parts) > 1:
                        chunk = parts[1]
                        if chunk.strip():
                            content_buffer += chunk
                            yield {"type": "content", "content": chunk}
                    yield {
                        "type": "thinking_done",
                        "content": thinking_buffer,
                    }
                    continue

                if in_thinking:
                    thinking_buffer += chunk
                    yield {"type": "thinking", "content": chunk}
                else:
                    content_buffer += chunk
                    yield {"type": "content", "content": chunk}

            # 流结束
            yield {"type": "done", "content": ""}

        except Exception as e:
            yield {
                "type": "error",
                "content": "",
                "error": f"流式输出异常: {str(e)}",
            }