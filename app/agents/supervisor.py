from app.agents.resume_agent import ResumeAgent
from app.agents.jd_agent import JDAgent
from app.agents.match_agent import MatchAgent
from app.agents.rewrite_agent import RewriteAgent
from app.agents.review_agent import ReviewAgent
from app.agents.interview_agent import InterviewAgent

class SupervisorAgent:
    """主管 Agent：负责任务编排，统一调度多个专业 Agent"""

    def __init__(self):
        self.resume_agent = ResumeAgent()
        self.jd_agent = JDAgent()
        self.match_agent = MatchAgent()
        self.rewrite_agent = RewriteAgent()
        self.review_agent = ReviewAgent()
        self.interview_agent = InterviewAgent()

    def analyze_resume_jd(self, resume_text: str, jd_text: str) -> dict:
        # 1. 调用简历解析 Agent
        resume_result = self.resume_agent.run(resume_text)

        # 2. 调用 JD 分析 Agent
        jd_result = self.jd_agent.run(jd_text)

        # 3. 调用匹配度分析 Agent
        match_result = self.match_agent.run(
            resume_skills=resume_result["resume_skills"],
            jd_skills=jd_result["jd_skills"]
        )

        # 4. 调用简历改写 Agent
        rewrite_result = self.rewrite_agent.run(
            resume_text=resume_text,
            jd_text=jd_text,
            matched_skills=match_result["matched_skills"],
            missing_skills=match_result["missing_skills"]
        )

        # 5. 调用结果校验 Agent
        review_result = self.review_agent.run(
            resume_text=resume_text,
            jd_text=jd_text,
            rewritten_project=rewrite_result["rewritten_project"],
            resume_skills=match_result["resume_skills"],
            jd_skills=match_result["jd_skills"]
        )
        interview_result = self.interview_agent.run(
            resume_skills=match_result["resume_skills"],
            jd_skills=match_result["jd_skills"],
            matched_skills=match_result["matched_skills"],
            missing_skills=match_result["missing_skills"],
            match_score=match_result["match_score"]
        )

        # 6. 汇总完整结果
        return {
            "workflow": [
                resume_result["agent_name"],
                jd_result["agent_name"],
                match_result["agent_name"],
                rewrite_result["agent_name"],
                review_result["agent_name"]
            ],
            "agent_summaries": [
                resume_result["summary"],
                jd_result["summary"],
                match_result["summary"],
                rewrite_result["summary"],
                review_result["summary"]
            ],
            "match_score": match_result["match_score"],
            "resume_skills": match_result["resume_skills"],
            "jd_skills": match_result["jd_skills"],
            "matched_skills": match_result["matched_skills"],
            "missing_skills": match_result["missing_skills"],
            "suggestions": match_result["suggestions"],
            "rewritten_project": rewrite_result["rewritten_project"],
            "rewrite_tips": rewrite_result["rewrite_tips"],
            "review_status": review_result["review_status"],
            "risk_items": review_result["risk_items"],
            "safe_suggestions": review_result["safe_suggestions"]
        }