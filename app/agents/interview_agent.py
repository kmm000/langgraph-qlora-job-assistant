import re
from app.tools.llm_client import call_ollama



class InterviewAgent:
    """面试辅导 Agent：根据简历技能、JD 要求和匹配结果，调用本地大模型生成面试题"""

    def run(
        self,
        resume_skills: list[str],
        jd_skills: list[str],
        matched_skills: list[str],
        missing_skills: list[str],
        match_score: float
    ) -> dict:
        interview_questions = self._generate_questions(
            resume_skills=resume_skills,
            jd_skills=jd_skills,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            match_score=match_score
        )

        answer_tips = self._generate_answer_tips(
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            match_score=match_score
        )

        return {
            "agent_name": "InterviewAgent",
            "interview_questions": interview_questions,
            "answer_tips": answer_tips,
            "summary": f"已调用本地大模型生成 {len(interview_questions)} 道面试题。"
        }

    def _generate_questions(
        self,
        resume_skills: list[str],
        jd_skills: list[str],
        matched_skills: list[str],
        missing_skills: list[str],
        match_score: float
    ) -> list[str]:
        prompt = f"""
你是一个大模型应用开发实习岗位的技术面试官。

请根据候选人简历技能、岗位技能要求和匹配结果，生成 10 道中文面试题。

要求：
1. 问题要贴合大模型应用开发、Agent、RAG、FastAPI、LangChain、模型微调等方向。
2. 一部分问题围绕候选人已经掌握的技能追问。
3. 一部分问题围绕候选人缺失的技能考察了解程度。
4. 问题要适合实习生，不要太偏高级架构。
5. 只输出问题列表，每行一个问题。
6. 每个问题以数字编号开头。

【简历已有技能】
{resume_skills}

【岗位要求技能】
{jd_skills}

【匹配技能】
{matched_skills}

【缺失技能】
{missing_skills}

【岗位匹配度】
{match_score}%

请生成 10 道面试题：
"""

        result = call_ollama(prompt)

        if result.startswith("【LLM调用失败】") or not result:
            return self._fallback_questions(matched_skills, missing_skills)

        questions = []

        for line in result.splitlines():
            line = line.strip()

            if not line:
                continue

            # 去掉模型生成的编号，例如：
            # 1. 问题
            # 1、问题
            # 1) 问题
            # （1）问题
            cleaned_line = re.sub(
                r"^\s*[（(]?\d+[）)]?\s*[.、:：)]*\s*",
                "",
                line
            ).strip()

            if cleaned_line:
                questions.append(cleaned_line)

        return (
            questions[:10]
            if questions
            else self._fallback_questions(matched_skills, missing_skills)
        )

    def _fallback_questions(self, matched_skills: list[str], missing_skills: list[str]) -> list[str]:
        questions = []

        for skill in matched_skills[:5]:
            questions.append(f"请结合你的项目经历，说明你是如何使用 {skill} 的？")

        for skill in missing_skills[:5]:
            questions.append(f"岗位中提到了 {skill}，你目前了解多少？后续准备如何补齐？")

        questions.extend([
            "你做过的项目中，哪一部分最能体现你的工程开发能力？",
            "如果让你设计一个多 Agent 求职辅助系统，你会如何拆分 Agent 职责？",
            "你如何避免大模型在简历改写中产生虚构内容？",
            "FastAPI 在这个项目中承担什么作用？",
            "如果系统输出结果不稳定，你会从 Prompt、工具调用和数据校验哪些方面优化？"
        ])

        return questions[:10]

    def _generate_answer_tips(
        self,
        matched_skills: list[str],
        missing_skills: list[str],
        match_score: float
    ) -> list[str]:
        tips = []

        tips.append(f"当前岗位匹配度为 {match_score}%，回答时应优先突出已匹配技能。")

        if matched_skills:
            tips.append(f"重点展示这些技能的实际项目应用：{', '.join(matched_skills[:5])}。")

        if missing_skills:
            tips.append(f"对于暂未掌握的技能，如 {', '.join(missing_skills[:5])}，建议诚实说明了解程度，并补充学习计划。")

        tips.append("回答项目问题时建议使用 STAR 结构：背景、任务、行动、结果。")
        tips.append("回答大模型应用开发相关问题时，重点讲清楚：业务场景、技术架构、接口设计、Agent 分工和结果校验。")

        return tips