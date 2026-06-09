from app.rag.vector_store import job_vector_store


class RAGAgent:
    """检索岗位知识、简历案例和面试知识。"""

    def run(
        self,
        resume_text: str,
        jd_text: str,
        top_k: int = 4,
    ) -> dict:
        query = (
            f"候选人简历：\n{resume_text}\n\n"
            f"目标岗位 JD：\n{jd_text}"
        )

        search_results = job_vector_store.search(
            query=query,
            top_k=top_k,
        )

        contexts = [
            item["document"]
            for item in search_results
        ]

        return {
            "agent_name": "RAGAgent",
            "rag_contexts": contexts,
            "rag_results": search_results,
            "summary": (
                f"已从 ChromaDB 检索到 "
                f"{len(search_results)} 条相关知识。"
            ),
        }