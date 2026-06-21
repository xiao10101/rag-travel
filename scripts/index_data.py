"""
数据索引入口脚本

用法：
    python scripts/index_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.text_splitter import split_pages
from app.retrieval.pdf_loader import load_pdf
from app.core.milvus_manager import MilvusManager
from app.retrieval.hybrid_retriever import HybridRetriever
from app.config import UPLOADS_DIR
import os


def main():
    manager = MilvusManager()

    # 1. 逐页读取 PDF
    pdf_path = os.path.join(UPLOADS_DIR, "docs.pdf")
    pages = load_pdf(pdf_path)

    # 2. 逐页分块，保留页码
    chunks = split_pages(pages)
    print("chunk 数量:", len(chunks))

    # 插入 Milvus 向量库（带页码和窗口上下文）
    texts = [c["text"] for c in chunks]
    metadatas = [{"page_num": c["page_num"], "window": c.get("window", c["text"])} for c in chunks]
    result = manager.insert(texts, metadatas)
    print("Milvus 插入成功")

    # 3. 构建 BM25 全文索引（同样基于这些 chunk）
    retriever = HybridRetriever()
    retriever.build_index(chunks)
    print("BM25 索引构建完成")


if __name__ == "__main__":
    main()