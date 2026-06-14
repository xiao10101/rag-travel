from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pages(pages: list[dict]) -> list[dict]:
    """对每页文本分别分块，返回每个 chunk 及其来源页码"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []
    for page in pages:
        page_texts = splitter.split_text(page["text"])
        for text in page_texts:
            chunks.append({
                "text": text,
                "page_num": page["page_num"]
            })

    return chunks