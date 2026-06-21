"""
RAGAS 评估模块

评估指标：
  - Context Precision（检索精度）
  - Context Recall（检索召回）
  - Faithfulness（生成忠实度）
  - Answer Relevancy（回答相关性）
  - 端到端延迟（P50 / P95 / P99）

使用方式：
  python scripts/run_eval.py                    # 评估全部 golden_data
  python scripts/run_eval.py --limit 5          # 仅评估前 5 条（快速验证）
  python scripts/run_eval.py --output result.json  # 结果写入 JSON 文件
"""

import time
import json
import argparse
import warnings
import numpy as np
from typing import Optional

from app.config import DASHSCOPE_API_KEY, GOLDEN_DATA_FILE
from app.core.llm import LLMService
from app.retrieval.hybrid_retriever import HybridRetriever

# 抑制 RAGAS 旧版导入路径的 deprecation warning（v1.0 前仍可用）
warnings.filterwarnings("ignore", message=".*ragas.metrics.*deprecated.*")

# ---- RAGAS 核心依赖 ----
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

# ---- 配置 RAGAS 评估器 LLM（使用 DashScope OpenAI 兼容接口） ----
from ragas.llms import llm_factory
from openai import OpenAI

_evaluator_llm = llm_factory(
    "qwen-plus",
    client=OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)

# RAGAS 0.4.x 通过 evaluate() 的 llm 参数统一注入评估器 LLM


# ============================================================
# 核心评估逻辑
# ============================================================

def run_single_query(
    question: str,
    retriever: HybridRetriever,
) -> dict:
    """执行单条 RAG pipeline：rewrite → retrieve → generate，返回完整数据"""
    latencies = {}

    # Step 1: Query Rewrite
    t0 = time.perf_counter()
    rewritten_query = LLMService.rewrite_query(question)
    latencies["rewrite"] = time.perf_counter() - t0

    # Step 2: Hybrid Retrieve
    t0 = time.perf_counter()
    raw_contexts = retriever.search(rewritten_query)
    latencies["retrieve"] = time.perf_counter() - t0

    # 提取纯文本列表供 RAGAS 使用
    context_texts = [ctx["text"] for ctx in raw_contexts]

    # Step 3: LLM Generate
    t0 = time.perf_counter()
    answer = LLMService.chat(question=question, contexts=raw_contexts)
    latencies["generate"] = time.perf_counter() - t0

    latencies["total"] = sum(latencies.values())

    return {
        "question": question,
        "rewritten_query": rewritten_query,
        "contexts": context_texts,          # list[str]，RAGAS 所需格式
        "answer": answer,
        "latencies": latencies,
    }


def build_evaluation_dataset(results: list[dict], golden_data: list[dict]) -> Dataset:
    """将 pipeline 运行结果 + golden_data 构建为 RAGAS 所需的 HuggingFace Dataset"""
    records = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for r, g in zip(results, golden_data):
        records["question"].append(r["question"])
        records["answer"].append(r["answer"])
        records["contexts"].append(r["contexts"])
        records["ground_truth"].append(g["ground_truth"])

    return Dataset.from_dict(records)


def compute_latency_percentiles(latencies: list[float]) -> dict:
    """计算端到端延迟的 P50 / P95 / P99"""
    arr = np.array(latencies)
    return {
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "mean": round(float(np.mean(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
    }


def print_results(ragas_scores: dict, latency_stats: dict):
    """格式化打印评估结果"""
    print("\n" + "=" * 60)
    print("   RAGAS 评估结果")
    print("=" * 60)

    metric_labels = {
        "context_precision": "Context Precision（检索精度）",
        "context_recall":    "Context Recall（检索召回）",
        "faithfulness":      "Faithfulness（生成忠实度）",
        "answer_relevancy":  "Answer Relevancy（回答相关性）",
    }

    for key, label in metric_labels.items():
        val = ragas_scores.get(key, "N/A")
        if isinstance(val, (int, float)):
            print(f"   {label:.<36s} {val:.4f}")
        else:
            print(f"   {label:.<36s} {val}")

    print("\n" + "-" * 60)
    print("   端到端延迟（End-to-End Latency）")
    print("-" * 60)
    print(f"   P50  (中位数):   {latency_stats['p50']}s")
    print(f"   P95:            {latency_stats['p95']}s")
    print(f"   P99:            {latency_stats['p99']}s")
    print(f"   Mean (平均):     {latency_stats['mean']}s")
    print(f"   Min / Max:       {latency_stats['min']}s / {latency_stats['max']}s")
    print("=" * 60 + "\n")


def run_evaluation(
    golden_path: str = GOLDEN_DATA_FILE,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
):
    """主评估流程"""
    # 1. 加载 golden data
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    if limit:
        golden_data = golden_data[:limit]

    print(f"加载 {len(golden_data)} 条 golden data")

    # 2. 初始化检索器
    retriever = HybridRetriever()

    # 3. 逐条运行 pipeline
    print("\n开始逐条评估...")
    results = []
    for i, item in enumerate(golden_data, 1):
        question = item["question"]
        print(f"  [{i}/{len(golden_data)}] {question[:50]}...", end=" ", flush=True)
        try:
            result = run_single_query(question, retriever)
            results.append(result)
            print(f"✓ ({result['latencies']['total']:.2f}s)")
        except Exception as e:
            print(f"✗ 失败: {e}")

    if not results:
        print("没有成功的评估结果，退出")
        return

    print(f"\n成功评估 {len(results)}/{len(golden_data)} 条")

    # 4. 构建 RAGAS Dataset 并计算
    print("\n计算 RAGAS 指标...")
    eval_dataset = build_evaluation_dataset(results, golden_data[:len(results)])

    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=_evaluator_llm,
    )

    ragas_scores = ragas_result._repr_dict

    # 5. 计算延迟百分位
    total_latencies = [r["latencies"]["total"] for r in results]
    latency_stats = compute_latency_percentiles(total_latencies)

    # 6. 输出
    print_results(ragas_scores, latency_stats)

    # 7. 可选持久化
    if output_path:
        output = {
            "ragas_scores": ragas_scores,
            "latency_stats": latency_stats,
            "per_query": [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "latency_total": r["latencies"]["total"],
                    "latency_breakdown": r["latencies"],
                }
                for r in results
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"结果已写入 {output_path}")


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS 评估脚本")
    parser.add_argument("--limit", type=int, default=None, help="仅评估前 N 条数据")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 文件路径")
    parser.add_argument("--golden", type=str, default=GOLDEN_DATA_FILE, help="golden data 文件路径")
    args = parser.parse_args()

    run_evaluation(
        golden_path=args.golden,
        limit=args.limit,
        output_path=args.output,
    )