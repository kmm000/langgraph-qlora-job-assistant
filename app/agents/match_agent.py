from app.tools.hybrid_match_score import (
    calculate_hybrid_match,
)


class MatchAgent:
    """
    混合匹配 Agent：
    结合技能匹配、整体语义相似度和项目相关性。
    """

    def run(
        self,
        resume_text: str,
        jd_text: str,
        resume_skills: list[str],
        jd_skills: list[str],
    ) -> dict:
        result = calculate_hybrid_match(
            resume_text=resume_text,
            jd_text=jd_text,
            resume_skills=resume_skills,
            jd_skills=jd_skills,
        )

        suggestions = self._generate_suggestions(
            missing_skills=result[
                "missing_skills"
            ],
            skill_score=result[
                "skill_score"
            ],
            semantic_score=result[
                "semantic_score"
            ],
            project_score=result[
                "project_score"
            ],
        )

        result.update(
            {
                "agent_name": "MatchAgent",
                "suggestions": suggestions,
                "summary": (
                    "已完成混合岗位匹配评估："
                    f"综合匹配度 {result['match_score']}%，"
                    f"技能匹配 {result['skill_score']}%，"
                    f"语义相似度 {result['semantic_score']}%，"
                    f"项目相关性 {result['project_score']}%。"
                ),
            }
        )

        return result

    def _generate_suggestions(
        self,
        missing_skills: list[str],
        skill_score: float,
        semantic_score: float,
        project_score: float,
    ) -> list[str]:
        suggestions = []

        if missing_skills:
            suggestions.append(
                "岗位中尚未匹配的技能包括："
                + "、".join(missing_skills[:6])
                + "。建议学习后在真实项目中补充。"
            )

        if skill_score < 50:
            suggestions.append(
                "技能关键词匹配较低，建议在专业技能和项目经历中明确体现已实际使用的岗位相关技术。"
            )

        if semantic_score < 50:
            suggestions.append(
                "简历整体内容与岗位语义关联较弱，建议调整项目描述，使业务场景和岗位职责更加贴合。"
            )

        if project_score < 50:
            suggestions.append(
                "项目经历与岗位要求的相关性不足，建议优先展示与目标岗位最接近的项目。"
            )

        if not suggestions:
            suggestions.append(
                "当前简历与岗位匹配度较高，建议进一步补充可量化且可验证的项目结果。"
            )

        return suggestions