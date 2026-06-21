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
你是一个旅游文档问答助手。请根据以下文档片段回答用户问题。

如果文档片段中包含答案，请准确回答；如果信息不足，请明确说明。

文档片段：
{formatted_context}

用户问题：
{question}

回答：
"""
        response = Generation.call(
            model="qwen-plus",
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
            result_format="message"
        )

        if response.status_code != HTTPStatus.OK:
            return f"生成回答失败: {response.message}"

        return response.output.choices[0].message.content.strip()

    @staticmethod
    def chat_stream(question: str, contexts: list[dict]) -> Generator[dict, None, None]:
        """流式生成回答，contexts 为 [{"text": ..., "page_num": ...}, ...]"""
        context_lines = []
        for i, ctx in enumerate(contexts, start=1):
            context_lines.append(
                f"[片段 {i} (第 {ctx['page_num']} 页)]\n{ctx['text']}"
            )
        formatted_context = "\n\n".join(context_lines)

        prompt = f"""
你是一个旅游文档问答助手。请根据以下文档片段回答用户问题。

如果文档片段中包含答案，请准确回答；如果信息不足，请明确说明。

文档片段：
{formatted_context}

用户问题：
{question}

回答：
"""
        try:
            responses = Generation.call(
                model="qwen-plus",
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
                stream=True,
                incremental_output=True,
            )

            for resp in responses:
                if resp.status_code != HTTPStatus.OK:
                    yield {"type": "error", "error": resp.message}
                    return

                reasoning_content = resp.output.choices[0].message.reasoning_content
                content = resp.output.choices[0].message.content

                if reasoning_content:
                    yield {"type": "thinking", "content": reasoning_content}
                if content:
                    yield {"type": "content", "content": content}

            yield {"type": "done"}

        except Exception as e:
            yield {"type": "error", "error": str(e)}