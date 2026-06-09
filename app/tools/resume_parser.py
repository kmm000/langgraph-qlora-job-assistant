import re


COMMON_SKILLS = [
    "Python", "PyTorch", "TensorFlow", "Scikit-learn", "Pandas", "NumPy",
    "FastAPI", "Flask", "Django", "Streamlit",
    "LangChain", "LangGraph", "LlamaIndex", "RAG", "Agent", "Prompt",
    "FAISS", "Chroma", "Milvus", "向量数据库",
    "Transformer", "YOLOv5", "深度学习", "机器学习", "迁移学习",
    "LoRA", "QLoRA", "模型微调", "大模型", "LLM",
    "Docker", "Linux", "Git", "MySQL", "Redis", "SQL",
    "数据处理", "模型训练", "结果可视化"
]


def load_text(file_path: str) -> str:
    """读取 txt 文本文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_skills(text: str) -> list[str]:
    """从简历文本中提取技能关键词"""
    found_skills = []

    lower_text = text.lower()

    for skill in COMMON_SKILLS:
        if skill.lower() in lower_text:
            found_skills.append(skill)

    return sorted(list(set(found_skills)))


def parse_resume(file_path: str) -> dict:
    """解析简历，返回结构化结果"""
    text = load_text(file_path)
    skills = extract_skills(text)

    return {
        "raw_text": text,
        "skills": skills
    }


if __name__ == "__main__":
    result = parse_resume("../../examples/resume.txt")
    print(result)