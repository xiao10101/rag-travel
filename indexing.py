from text_splitter import split_pages
from pdf_loader import load_pdf
from milvus_manager import MilvusManager


def main():
    manager = MilvusManager()

    # 1. 逐页读取 PDF
    pages = load_pdf("./uploads/docs.pdf")

    # 2. 逐页分块，保留页码
    chunks = split_pages(pages)
    print("chunk 数量:", len(chunks))

    # 插入数据库（带页码信息）
    texts = [c["text"] for c in chunks]
    metadatas = [{"page_num": c["page_num"]} for c in chunks]
    result = manager.insert(texts, metadatas)

    print("插入成功")
    print(result)


if __name__ == "__main__":
    main()