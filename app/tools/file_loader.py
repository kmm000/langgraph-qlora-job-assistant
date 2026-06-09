from pathlib import Path
from docx import Document
from pypdf import PdfReader


def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    texts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)

    return "\n".join(texts)


def load_resume_file(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()

    if suffix == ".txt":
        return read_txt(file_path)

    if suffix == ".docx":
        return read_docx(file_path)

    if suffix == ".pdf":
        return read_pdf(file_path)

    raise ValueError("暂不支持该文件格式，请上传 txt、docx 或 pdf 文件。")