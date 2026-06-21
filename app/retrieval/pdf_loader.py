from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    """逐页读取 PDF，返回每页的页码和文本"""
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text.strip():
            pages.append({"page_num": i, "text": text})

    return pages