"""
RAGAS 评估模块

评估指标：
  - Context Precision（检索精度）
  - Context Recall（检索召回）
  - Faithfulness（生成忠实度）
  - Answer Relevancy（回答相关性）
  - 端到端延迟（P50 / P95 / P99）

使用方式：
  python scripts/run_eval.py                     # 评估全部 golden_data
  python scripts/run_eval.py --limit 5           # 仅评估前 5 条（快速验证）
  python scripts/run_eval.py --fast              # 快速模式（仅 precision + recall，跳过慢指标）
  python scripts/run_eval.py --parallel 4        # 并行执行 pipeline（4 线程）
  python scripts/run_eval.py --output result.json # 结果写入 JSON 文件
"""

import time
import json
import argparse
import warnings
import logging
import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import DASHSCOPE_API_KEY, GOLDEN_DATA_FILE
from app.core.llm import LLMService
from app.retrieval.hybrid_retriever import HybridRetriever

# 抑制 RAGAS 旧版导入路径的 deprecation warning
warnings.filterwarnings("ignore", message=".*ragas.metrics.*deprecated.*")
# 抑制 DashScope 不支持 n 参数的 warning（RAGAS 内部请求 n=3，但 API 只返回 1）
warnings.filterwarnings("ignore", message=".*returned.*generations.*instead of requested.*")
logging.getLogger("ragas").setLevel(logging.WARNING)

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

# 默认使用 qwen-turbo 作为评估 LLM（评估只需判断语义，不需要最强模型，速度更快成本更低）
# 可通过 --eval-model 参数覆盖
EVAL_LLM_MODEL = "qwen-turbo"

def _build_evaluator_llm(model: str = EVAL_LLM_MODEL):
    """构建 RAGAS 评估器 LLM"""
    return llm_factory(
        model,
        client=OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        max_tokens=2048,
    )

# 配置 Embeddings（用于 answer_relevancy 指标）
# 自定义适配器包装 DashScope 的 OpenAI 兼容接口
from ragas.embeddings.base import BaseRagasEmbeddings

class DashScopeEmbeddings(BaseRagasEmbeddings):
    """DashScope Embeddings 适配器"""
    
    def __init__(self, model: str, client: OpenAI):
        self.model = model
        self.client = client
    
    def embed_text(self, text: str) -> list[float]:
        """嵌入单个文本"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding
    
    def embed_query(self, text: str) -> list[float]:
        """嵌入查询文本（兼容 LangChain 接口）"""
        return self.embed_text(text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """嵌入多个文档"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    async def aembed_query(self, text: str) -> list[float]:
        """异步嵌入查询文本"""
        return self.embed_query(text)
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步嵌入多个文档"""
        return self.embed_documents(texts)

_evaluator_embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
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
    fast: bool = False,
    parallel: int = 1,
    eval_model: str = EVAL_LLM_MODEL,
    metrics: Optional[list[str]] = None,
):
    """主评估流程

    Args:
        golden_path: golden data 文件路径
        limit: 仅评估前 N 条
        output_path: 结果 JSON 输出路径
        fast: 快速模式，仅计算 context_precision + context_recall（跳过耗时的 faithfulness 和 answer_relevancy）
        parallel: 并行执行 pipeline 的线程数（默认 1，串行）
        eval_model: RAGAS 评估器使用的 LLM 模型名
        metrics: 指定要计算的指标列表，如 ["context_precision", "context_recall"]
    """
    # 1. 加载 golden data
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    if limit:
        golden_data = golden_data[:limit]

    print(f"加载 {len(golden_data)} 条 golden data")

    # 2. 初始化检索器
    retriever = HybridRetriever()

    # 3. 运行 pipeline（支持并行）
    if parallel > 1:
        print(f"\n开始并行评估（{parallel} 线程）...")
        results = _run_pipeline_parallel(golden_data, retriever, parallel)
    else:
        print("\n开始逐条评估...")
        results = _run_pipeline_sequential(golden_data, retriever)

    if not results:
        print("没有成功的评估结果，退出")
        return

    print(f"\n成功评估 {len(results)}/{len(golden_data)} 条")

    # 4. 确定要计算的指标
    all_metrics_map = {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }

    if metrics:
        # 用户指定了具体指标
        selected_metrics = [all_metrics_map[m] for m in metrics if m in all_metrics_map]
    elif fast:
        # 快速模式：跳过最耗时的两个指标
        selected_metrics = [context_precision, context_recall]
        print("快速模式：仅计算 Context Precision + Context Recall")
    else:
        selected_metrics = [context_precision, context_recall, faithfulness, answer_relevancy]

    # 5. 构建 RAGAS Dataset 并计算
    print(f"\n计算 RAGAS 指标（评估模型: {eval_model}）...")
    eval_dataset = build_evaluation_dataset(results, golden_data[:len(results)])
    evaluator_llm = _build_evaluator_llm(eval_model)

    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=selected_metrics,
        llm=evaluator_llm,
        embeddings=_evaluator_embeddings,
    )

    ragas_scores = ragas_result._repr_dict

    # 6. 计算延迟百分位
    total_latencies = [r["latencies"]["total"] for r in results]
    latency_stats = compute_latency_percentiles(total_latencies)

    # 7. 输出
    print_results(ragas_scores, latency_stats)

    # 8. 可选持久化
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


def _run_pipeline_sequential(
    golden_data: list[dict],
    retriever: HybridRetriever,
) -> list[dict]:
    """串行执行 pipeline"""
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
    return results


def _run_pipeline_parallel(
    golden_data: list[dict],
    retriever: HybridRetriever,
    max_workers: int,
) -> list[dict]:
    """并行执行 pipeline（多线程）"""
    results = []

    def _run_one(idx: int, item: dict) -> tuple[int, dict | None, str | None]:
        question = item["question"]
        try:
            result = run_single_query(question, retriever)
            return idx, result, None
        except Exception as e:
            return idx, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, i, item): i
            for i, item in enumerate(golden_data)
        }
        for future in as_completed(futures):
            idx, result, error = future.result()
            question = golden_data[idx]["question"]
            if error:
                print(f"  [{idx+1}/{len(golden_data)}] {question[:50]}... ✗ 失败: {error}")
            else:
                results.append((idx, result))
                print(f"  [{idx+1}/{len(golden_data)}] {question[:50]}... ✓ ({result['latencies']['total']:.2f}s)")

    # 按原始顺序排列
    results.sort(key=lambda x: x[0])
    return [r for _, r in results]


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS 评估脚本")
    parser.add_argument("--limit", type=int, default=None, help="仅评估前 N 条数据")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 文件路径")
    parser.add_argument("--golden", type=str, default=GOLDEN_DATA_FILE, help="golden data 文件路径")
    parser.add_argument("--fast", action="store_true", help="快速模式：仅计算 precision + recall，跳过耗时的 faithfulness 和 answer_relevancy")
    parser.add_argument("--parallel", type=int, default=1, help="并行执行 pipeline 的线程数（默认 1）")
    parser.add_argument("--eval-model", type=str, default=EVAL_LLM_MODEL, help=f"RAGAS 评估器 LLM 模型（默认: {EVAL_LLM_MODEL}）")
    parser.add_argument("--metrics", type=str, nargs="*", default=None,
                        help="指定要计算的指标，如: context_precision context_recall")
    args = parser.parse_args()

    run_evaluation(
        golden_path=args.golden,
        limit=args.limit,
        output_path=args.output,
        fast=args.fast,
        parallel=args.parallel,
        eval_model=args.eval_model,
        metrics=args.metrics,
    )