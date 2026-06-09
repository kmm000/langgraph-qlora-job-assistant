import hashlib
import re


def normalize_text(text: str) -> str:
    """规范化文本，减少换行和多余空格造成的哈希差异。"""
    text = text.strip()
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def calculate_document_hash(document_text: str) -> str:
    """根据规范化后的文档文本计算 SHA256。"""
    normalized_text = normalize_text(document_text)

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """按字符长度切分文本，并保留重叠区域。"""
    normalized_text = normalize_text(text)

    if not normalized_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap 必须大于等于 0，且小于 chunk_size。"
        )

    chunks = []
    start = 0
    text_length = len(normalized_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = normalized_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def build_chunk_id(
    document_hash: str,
    chunk_index: int,
) -> str:
    """根据文档哈希和分块序号生成稳定 ID。"""
    return f"{document_hash}_chunk_{chunk_index:04d}"