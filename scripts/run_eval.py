"""
评估入口脚本

用法：
    python scripts/run_eval.py                         # 评估全部 golden_data
    python scripts/run_eval.py --limit 5               # 仅评估前 5 条（快速验证）
    python scripts/run_eval.py --fast                  # 快速模式（仅 precision + recall）
    python scripts/run_eval.py --parallel 4            # 4 线程并行执行 pipeline
    python scripts/run_eval.py --fast --parallel 4     # 组合使用：最快速度
    python scripts/run_eval.py --output result.json    # 结果写入 JSON 文件
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from app.evaluation import run_evaluation
from app.config import GOLDEN_DATA_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS 评估脚本")
    parser.add_argument("--limit", type=int, default=None, help="仅评估前 N 条数据")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 文件路径")
    parser.add_argument("--golden", type=str, default=GOLDEN_DATA_FILE, help="golden data 文件路径")
    parser.add_argument("--fast", action="store_true", help="快速模式：仅计算 precision + recall，跳过耗时的 faithfulness 和 answer_relevancy")
    parser.add_argument("--parallel", type=int, default=1, help="并行执行 pipeline 的线程数（默认 1）")
    parser.add_argument("--eval-model", type=str, default=None, help="RAGAS 评估器 LLM 模型（默认: qwen-turbo）")
    parser.add_argument("--metrics", type=str, nargs="*", default=None,
                        help="指定要计算的指标，如: context_precision context_recall")
    args = parser.parse_args()

    kwargs = {
        "golden_path": args.golden,
        "limit": args.limit,
        "output_path": args.output,
        "fast": args.fast,
        "parallel": args.parallel,
        "metrics": args.metrics,
    }
    if args.eval_model:
        kwargs["eval_model"] = args.eval_model

    run_evaluation(**kwargs)