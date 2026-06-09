from typing import Sequence

import requests


OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "bge-m3:latest"


class OllamaEmbeddingClient:
    """调用 Ollama bge-m3 生成文本向量。"""

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        base_url: str = OLLAMA_EMBED_URL,
    ):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        if not cleaned_texts:
            return []

        response = requests.post(
            self.base_url,
            json={
                "model": self.model,
                "input": cleaned_texts,
            },
            timeout=180,
        )

        response.raise_for_status()
        data = response.json()

        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError("Ollama 没有返回 embeddings。")

        if len(embeddings) != len(cleaned_texts):
            raise RuntimeError(
                f"向量数量与文本数量不一致："
                f"{len(embeddings)} != {len(cleaned_texts)}"
            )

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_documents([text])

        if not embeddings:
            raise ValueError("查询文本不能为空。")

        return embeddings[0]


embedding_client = OllamaEmbeddingClient()