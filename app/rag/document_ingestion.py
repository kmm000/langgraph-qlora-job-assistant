from typing import Any

from app.rag.document_utils import (
    build_chunk_id,
    calculate_document_hash,
    split_text,
)
from app.rag.vector_store import job_vector_store


def ingest_document(
    document_text: str,
    filename: str,
    category: str = "uploaded_document",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    force_update: bool = False,
) -> dict[str, Any]:
    """
    将文档切分、向量化并写入 ChromaDB。

    force_update=False：
        已存在时直接跳过。

    force_update=True：
        删除旧分块后重新写入。
    """
    if not document_text or not document_text.strip():
        raise ValueError("文档内容不能为空。")

    document_hash = calculate_document_hash(document_text)

    exists = job_vector_store.document_exists(
        document_hash=document_hash
    )

    if exists and not force_update:
        existing = job_vector_store.get_document_by_hash(
            document_hash=document_hash
        )

        return {
            "status": "already_exists",
            "message": "该文档已存在于向量库中，已跳过重复导入。",
            "filename": filename,
            "document_hash": document_hash,
            "chunk_count": len(existing.get("ids", [])),
        }

    deleted_count = 0

    if exists and force_update:
        deleted_count = job_vector_store.delete_document(
            document_hash=document_hash
        )

    chunks = split_text(
        text=document_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError("文档切分后没有产生有效文本块。")

    ids = [
        build_chunk_id(
            document_hash=document_hash,
            chunk_index=index,
        )
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "filename": filename,
            "document_hash": document_hash,
            "chunk_index": index,
            "chunk_count": len(chunks),
            "category": category,
            "source": "uploaded_file",
        }
        for index in range(len(chunks))
    ]

    # add_documents 内部使用 collection.upsert
    inserted_count = job_vector_store.add_documents(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )

    return {
        "status": "inserted",
        "message": "文档已成功写入向量库。",
        "filename": filename,
        "document_hash": document_hash,
        "chunk_count": inserted_count,
        "deleted_old_chunks": deleted_count,
    }