import math
import re
from typing import Any

from app.rag.embedding_client import embedding_client


DEFAULT_WEIGHTS = {
    "skill": 0.50,
    "semantic": 0.30,
    "project": 0.20,
}


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """计算两个向量的余弦相似度。"""

    if not vector_a or not vector_b:
        return 0.0

    if len(vector_a) != len(vector_b):
        raise ValueError("两个向量的维度不一致。")

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(value * value for value in vector_a)
    )

    norm_b = math.sqrt(
        sum(value * value for value in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)

    # 限制到 0～1
    return max(0.0, min(1.0, similarity))


def normalize_skill(skill: str) -> str:
    """统一技能名称，减少大小写和符号差异。"""

    return re.sub(
        r"[\s_\-+/]",
        "",
        skill.strip().lower(),
    )


def calculate_skill_score(
    resume_skills: list[str],
    jd_skills: list[str],
) -> dict[str, Any]:
    """计算技能精确匹配分。"""

    normalized_resume = {
        normalize_skill(skill): skill
        for skill in resume_skills
        if skill and skill.strip()
    }

    normalized_jd = {
        normalize_skill(skill): skill
        for skill in jd_skills
        if skill and skill.strip()
    }

    if not normalized_jd:
        return {
            "score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
        }

    matched_keys = (
        set(normalized_resume.keys())
        & set(normalized_jd.keys())
    )

    missing_keys = (
        set(normalized_jd.keys())
        - set(normalized_resume.keys())
    )

    matched_skills = sorted(
        normalized_jd[key]
        for key in matched_keys
    )

    missing_skills = sorted(
        normalized_jd[key]
        for key in missing_keys
    )

    score = (
        len(matched_keys)
        / len(normalized_jd)
        * 100
    )

    return {
        "score": round(score, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }


def calculate_semantic_score(
    resume_text: str,
    jd_text: str,
) -> float:
    """计算完整简历和 JD 的语义相似度。"""

    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    embeddings = embedding_client.embed_documents(
        [
            resume_text,
            jd_text,
        ]
    )

    similarity = cosine_similarity(
        embeddings[0],
        embeddings[1],
    )

    return round(similarity * 100, 2)


def split_resume_projects(
    resume_text: str,
) -> list[str]:
    """
    将简历按段落切分，筛选可能的项目经历。

    第一版不依赖复杂解析器，优先保留长度较长、
    包含技术或项目关键词的段落。
    """

    normalized_text = resume_text.replace(
        "\r\n",
        "\n",
    )

    raw_sections = re.split(
        r"\n\s*\n|(?=\n\d+[.、])",
        normalized_text,
    )

    project_keywords = {
        "项目",
        "开发",
        "系统",
        "模型",
        "实现",
        "构建",
        "设计",
        "训练",
        "预测",
        "部署",
        "接口",
        "agent",
        "rag",
        "fastapi",
        "pytorch",
        "langgraph",
        "yolo",
    }

    projects = []

    for section in raw_sections:
        section = section.strip()

        if len(section) < 20:
            continue

        lower_section = section.lower()

        if any(
            keyword in lower_section
            for keyword in project_keywords
        ):
            projects.append(section)

    # 如果没识别出项目，退化为使用完整简历
    if not projects and resume_text.strip():
        projects.append(resume_text.strip())

    return projects[:10]


def calculate_project_score(
    resume_text: str,
    jd_text: str,
    top_n: int = 2,
) -> dict[str, Any]:
    """计算项目段落和 JD 的相关性。"""

    projects = split_resume_projects(resume_text)

    if not projects or not jd_text.strip():
        return {
            "score": 0.0,
            "project_scores": [],
        }

    texts = projects + [jd_text]
    embeddings = embedding_client.embed_documents(texts)

    jd_embedding = embeddings[-1]
    project_embeddings = embeddings[:-1]

    project_scores = []

    for project, project_embedding in zip(
        projects,
        project_embeddings,
    ):
        similarity = cosine_similarity(
            project_embedding,
            jd_embedding,
        )

        project_scores.append(
            {
                "project_preview": project[:120],
                "score": round(
                    similarity * 100,
                    2,
                ),
            }
        )

    project_scores.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected_scores = [
        item["score"]
        for item in project_scores[:top_n]
    ]

    final_score = (
        sum(selected_scores)
        / len(selected_scores)
        if selected_scores
        else 0.0
    )

    return {
        "score": round(final_score, 2),
        "project_scores": project_scores,
    }


def calculate_hybrid_match(
    resume_text: str,
    jd_text: str,
    resume_skills: list[str],
    jd_skills: list[str],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """计算关键词与语义结合的综合匹配度。"""

    actual_weights = weights or DEFAULT_WEIGHTS

    skill_result = calculate_skill_score(
        resume_skills=resume_skills,
        jd_skills=jd_skills,
    )

    # Embedding 服务异常时，退化为技能匹配
    try:
        semantic_score = calculate_semantic_score(
            resume_text=resume_text,
            jd_text=jd_text,
        )

        project_result = calculate_project_score(
            resume_text=resume_text,
            jd_text=jd_text,
        )

        embedding_available = True

    except Exception as exc:
        semantic_score = 0.0

        project_result = {
            "score": 0.0,
            "project_scores": [],
        }

        embedding_available = False
        embedding_error = str(exc)

    final_score = (
        skill_result["score"]
        * actual_weights["skill"]
        + semantic_score
        * actual_weights["semantic"]
        + project_result["score"]
        * actual_weights["project"]
    )

    result = {
        "match_score": round(final_score, 2),
        "skill_score": skill_result["score"],
        "semantic_score": semantic_score,
        "project_score": project_result["score"],
        "matched_skills": skill_result[
            "matched_skills"
        ],
        "missing_skills": skill_result[
            "missing_skills"
        ],
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "project_scores": project_result[
            "project_scores"
        ],
        "score_weights": actual_weights,
        "embedding_available": embedding_available,
    }

    if not embedding_available:
        result["embedding_error"] = embedding_error

        # Embedding 失败时只返回技能分，
        # 避免最终结果被两个 0 分严重拉低
        result["match_score"] = skill_result["score"]

    return result