from app.tools.resume_parser import extract_skills


class ResumeAgent:
    """简历解析 Agent：负责从简历文本中提取候选人技能"""

    def run(self, resume_text: str) -> dict:
        skills = extract_skills(resume_text)

        return {
            "agent_name": "ResumeAgent",
            "resume_skills": skills,
            "summary": f"已从简历中提取到 {len(skills)} 个技能关键词。"
        }