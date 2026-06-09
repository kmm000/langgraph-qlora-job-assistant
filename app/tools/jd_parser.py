from app.tools.resume_parser import COMMON_SKILLS, load_text


def extract_jd_skills(text: str) -> list[str]:
    """从 JD 文本中提取岗位技能关键词"""
    found_skills = []

    lower_text = text.lower()

    for skill in COMMON_SKILLS:
        if skill.lower() in lower_text:
            found_skills.append(skill)

    return sorted(list(set(found_skills)))


def parse_jd(file_path: str) -> dict:
    """解析岗位 JD，返回结构化结果"""
    text = load_text(file_path)
    skills = extract_jd_skills(text)

    return {
        "raw_text": text,
        "required_skills": skills
    }


if __name__ == "__main__":
    result = parse_jd("../../examples/jd.txt")
    print(result)