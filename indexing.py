from text_splitter import split_pages
from pdf_loader import load_pdf
from milvus_manager import MilvusManager
from hybrid_retriever import HybridRetriever


def main():
    manager = MilvusManager()

    # 1. 逐页读取 PDF
    pages = load_pdf("./uploads/docs.pdf")

    # 2. 逐页分块，保留页码
    chunks = split_pages(pages)
    print("chunk 数量:", len(chunks))

    # 插入 Milvus 向量库（带页码信息）
    texts = [c["text"] for c in chunks]
    metadatas = [{"page_num": c["page_num"]} for c in chunks]
    result = manager.insert(texts, metadatas)
    print("Milvus 插入成功")

    # 3. 构建 BM25 全文索引（同样基于这些 chunk）
    retriever = HybridRetriever()
    retriever.build_index(chunks)
    print("BM25 索引构建完成")


if __name__ == "__main__":
    main()