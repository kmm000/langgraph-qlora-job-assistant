from pathlib import Path
from typing import Any
import uuid

import chromadb

from app.rag.embedding_client import embedding_client


CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "job_assistant_knowledge"


class JobKnowledgeVectorStore:
    """求职辅助知识库的 ChromaDB 封装。"""

    def __init__(
        self,
        persist_directory: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "求职辅助系统知识库",
                "embedding_model": "bge-m3",
            },
        )

    def document_exists(
            self,
            document_hash: str,
    ) -> bool:
        """根据 document_hash 判断文档是否已经导入。"""
        result = self.collection.get(
            where={
                "document_hash": document_hash
            },
            include=["metadatas"],
        )

        return bool(result.get("ids"))

    def get_document_by_hash(
            self,
            document_hash: str,
    ) -> dict:
        """获取指定 document_hash 对应的分块记录。"""
        result = self.collection.get(
            where={
                "document_hash": document_hash
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        return result

    def delete_document(
            self,
            document_hash: str,
    ) -> int:
        """删除指定文档的全部 chunk。"""
        result = self.collection.get(
            where={
                "document_hash": document_hash
            },
            include=["metadatas"],
        )

        ids = result.get("ids", [])

        if not ids:
            return 0

        self.collection.delete(ids=ids)

        return len(ids)

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        cleaned_documents = [
            document.strip()
            for document in documents
            if document and document.strip()
        ]

        if not cleaned_documents:
            return 0

        if metadatas is None:
            metadatas = [
                {"source": "unknown"}
                for _ in cleaned_documents
            ]

        if len(metadatas) != len(cleaned_documents):
            raise ValueError("metadatas 数量必须与 documents 一致。")

        if ids is None:
            ids = [
                str(uuid.uuid4())
                for _ in cleaned_documents
            ]

        if len(ids) != len(cleaned_documents):
            raise ValueError("ids 数量必须与 documents 一致。")

        embeddings = embedding_client.embed_documents(
            cleaned_documents
        )

        self.collection.upsert(
            ids=ids,
            documents=cleaned_documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(cleaned_documents)

    def search(
        self,
        query: str,
        top_k: int = 4,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()

        if not query:
            return []

        query_embedding = embedding_client.embed_query(query)

        where = None

        if category:
            where = {
                "category": category
            }

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        search_results = []

        for doc_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            search_results.append(
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata or {},
                    "distance": round(float(distance), 6),
                }
            )

        return search_results

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(
            COLLECTION_NAME
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )


job_vector_store = JobKnowledgeVectorStore()