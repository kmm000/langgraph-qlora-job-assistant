import json
from pathlib import Path

from app.rag.vector_store import job_vector_store


KNOWLEDGE_FILES = [
    "knowledge/resume_cases.json",
    "knowledge/interview_knowledge.json",
    "knowledge/job_knowledge.json",
]


def load_json_file(file_path: str) -> list[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"知识库文件不存在：{path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"知识库文件必须是数组：{file_path}"
        )

    return data


def build_knowledge_base(
    reset: bool = False,
) -> int:
    if reset:
        job_vector_store.reset()

    documents = []
    metadatas = []
    ids = []

    for file_path in KNOWLEDGE_FILES:
        items = load_json_file(file_path)

        for item in items:
            document = (
                f"标题：{item.get('title', '')}\n"
                f"内容：{item.get('content', '')}"
            )

            documents.append(document)

            metadatas.append(
                {
                    "category": item.get(
                        "category",
                        "unknown",
                    ),
                    "title": item.get(
                        "title",
                        "",
                    ),
                    "source": file_path,
                }
            )

            ids.append(item["id"])

    added_count = job_vector_store.add_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    print(
        f"知识库构建完成，共写入 {added_count} 条数据。"
    )

    return added_count


if __name__ == "__main__":
    build_knowledge_base(reset=True)