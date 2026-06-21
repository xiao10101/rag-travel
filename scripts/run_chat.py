"""
交互式对话入口脚本

用法：
    python scripts/run_chat.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm import LLMService
from app.retrieval.hybrid_retriever import HybridRetriever


def print_thinking(content: str, end: str = ""):
    """打印思考过程（灰色样式）"""
    print(f"\033[90m{content}\033[0m", end=end, flush=True)


def print_content(content: str, end: str = ""):
    """打印回答内容（正常样式）"""
    print(content, end=end, flush=True)


def print_separator():
    """打印分隔线"""
    print("\n" + "─" * 60 + "\n")


def stream_chat(question: str, retriever: HybridRetriever):
    """执行流式对话：改写 → 检索 → 流式生成"""

    # Step 1: Rewrite — 改写查询
    print("\n🔍 正在分析问题...")
    rewritten_query = LLMService.rewrite_query(question)
    print(f"   ✅ 查询改写完成: {rewritten_query}\n")

    # Step 2: Retrieve — 混合检索
    print("📚 正在检索相关文档...")
    start_time = time.time()
    contexts = retriever.search(rewritten_query)
    elapsed = time.time() - start_time

    if not contexts:
        print("   ⚠️  未找到相关文档片段")
        return

    print(f"   ✅ 检索完成，找到 {len(contexts)} 个相关片段（耗时 {elapsed:.2f}s）")
    for i, ctx in enumerate(contexts, 1):
        preview = ctx["text"][:50].replace("\n", " ")
        print(f"      [{i}] 第 {ctx['page_num']} 页: {preview}...")

    # Step 3: Stream — 流式生成回答（带思考过程）
    print("\n💭 思考过程中...")
    print("─" * 40)

    thinking_started = False
    content_started = False
    full_answer = ""

    for chunk in LLMService.chat_stream(question=question, contexts=contexts):
        chunk_type = chunk["type"]

        if chunk_type == "thinking":
            if not thinking_started:
                thinking_started = True
                print("\n🤔 **思考过程**\n")
            print_thinking(chunk["content"], end="")

        elif chunk_type == "thinking_done":
            print_thinking("\n")

        elif chunk_type == "content":
            if not content_started:
                content_started = True
                print("\n\n📝 **回答**\n")
            print_content(chunk["content"], end="")
            full_answer += chunk["content"]

        elif chunk_type == "error":
            print(f"\n❌ 错误: {chunk['error']}")
            return

        elif chunk_type == "done":
            print("\n")
            break

    print_separator()
    return full_answer


def main():
    print("=" * 60)
    print("   🌍 RAG 旅游文档智能问答系统 (流式版)")
    print("=" * 60)
    print("\n输入问题开始对话，输入 'quit' 或 'exit' 退出\n")

    # 初始化检索器
    retriever = HybridRetriever()

    while True:
        try:
            question = input("❓ 你想问什么？> ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("\n👋 再见！")
                break

            stream_chat(question, retriever)

        except KeyboardInterrupt:
            print("\n\n👋 已中断，再见！")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            continue


if __name__ == "__main__":
    main()