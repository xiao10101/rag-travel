"""
评估入口脚本

用法：
    python scripts/run_eval.py                    # 评估全部 golden_data
    python scripts/run_eval.py --limit 5          # 仅评估前 5 条（快速验证）
    python scripts/run_eval.py --output result.json  # 结果写入 JSON 文件
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
    args = parser.parse_args()

    run_evaluation(
        golden_path=args.golden,
        limit=args.limit,
        output_path=args.output,
    )