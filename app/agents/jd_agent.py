from app.tools.jd_parser import extract_jd_skills


class JDAgent:
    """岗位 JD 分析 Agent：负责从岗位描述中提取岗位技能要求"""

    def run(self, jd_text: str) -> dict:
        required_skills = extract_jd_skills(jd_text)

        return {
            "agent_name": "JDAgent",
            "jd_skills": required_skills,
            "summary": f"已从岗位 JD 中提取到 {len(required_skills)} 个技能要求。"
        }