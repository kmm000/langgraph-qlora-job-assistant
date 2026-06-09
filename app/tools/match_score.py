def calculate_match_score(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """计算简历和 JD 的技能匹配度"""

    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched_skills = sorted(list(resume_set & jd_set))
    missing_skills = sorted(list(jd_set - resume_set))

    if len(jd_set) == 0:
        score = 0.0
    else:
        score = round(len(matched_skills) / len(jd_set) * 100, 2)

    return {
        "match_score": score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_skills": sorted(list(resume_set)),
        "jd_skills": sorted(list(jd_set))
    }


def generate_suggestions(missing_skills: list[str]) -> list[str]:
    """根据缺失技能生成建议"""
    suggestions = []

    if not missing_skills:
        return ["简历与岗位要求匹配度较高，可以进一步突出项目成果和量化指标。"]

    for skill in missing_skills:
        suggestions.append(f"建议补充或学习 {skill} 相关内容，并在项目经历中体现实际应用。")

    return suggestions