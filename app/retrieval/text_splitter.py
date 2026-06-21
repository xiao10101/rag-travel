from langchain_text_splitters import RecursiveCharacterTextSplitter


def _split_sentences(text: str) -> list[str]:
    """按中文/英文标点分句，保留分隔符"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=999999,
        separators=["。", "！", "？", "；", "\n", ". ", " "],
    )
    return [s.strip() for s in splitter.split_text(text) if s.strip()]


def split_pages(pages: list[dict]) -> list[dict]:
    """Sentence Window 分块策略

    将每页文本按句子切分，每个 chunk 为单句，
    同时附带前后 window_size 句作为上下文，用于检索时扩展。
    """
    window_size = 2  # 前后各 2 句
    chunks = []

    for page in pages:
        sentences = _split_sentences(page["text"])
        if not sentences:
            continue

        for i, sentence in enumerate(sentences):
            # 构建上下文窗口
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            window_context = "".join(sentences[start:end])

            chunks.append({
                "text": sentence,
                "page_num": page["page_num"],
                "window": window_context,
            })

    return chunks