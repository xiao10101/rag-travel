
from text_splitter import split_text
from pdf_loader import load_pdf
from milvus_manager import MilvusManager


def main():
    manager = MilvusManager()
    # 1. 读取 PDF
    text = load_pdf("./uploads/docs.pdf")

    # 2. chunk
    chunks = split_text(text)
    print("chunk 数量:", len(chunks))

    # 插入数据库
    result = manager.insert(chunks)

    print("插入成功")
    print(result)

if __name__ == "__main__":
    main()