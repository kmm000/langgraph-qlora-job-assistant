from app.tools.lora_rewrite_client import lora_rewrite_client


class RewriteAgent:
    """简历改写 Agent：调用 LoRA 微调模型生成项目经历优化版本"""

    def run(
        self,
        resume_text: str,
        jd_text: str,
        matched_skills: list[str],
        missing_skills: list[str],
        rag_contexts: list[str] | None = None,
    ) -> dict:
        try:
            rewritten_project = lora_rewrite_client.rewrite(
                resume_text=resume_text,
                jd_text=jd_text,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                rag_contexts=rag_contexts or [],
            )
        except Exception as e:
            rewritten_project = self._fallback_rewrite(
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                error_message=str(e),
            )

        rewrite_tips = self._generate_rewrite_tips(
            matched_skills=matched_skills,
            missing_skills=missing_skills,
        )

        return {
            "agent_name": "RewriteAgent",
            "rewritten_project": rewritten_project,
            "rewrite_tips": rewrite_tips,
            "summary": "已调用 LoRA 微调模型生成项目经历优化版本。",
        }

    def _fallback_rewrite(
        self,
        matched_skills: list[str],
        missing_skills: list[str],
        error_message: str,
    ) -> str:
        matched_text = "、".join(matched_skills) if matched_skills else "Python、项目开发、数据处理"
        missing_text = "、".join(missing_skills[:5]) if missing_skills else "岗位相关技术"

        return (
            f"【LoRA模型调用失败，已使用规则模板兜底】\n"
            f"错误信息：{error_message}\n\n"
            f"- 基于 {matched_text} 完成求职辅助系统开发，围绕简历解析、岗位 JD 分析、技能匹配度评估和优化建议生成等场景，构建完整业务流程。\n"
            f"- 设计模块化分析流程，实现从文本输入、关键词提取、匹配度计算到结果展示的完整闭环。\n"
            f"- 针对岗位中涉及的 {missing_text} 等能力要求，进一步扩展 Agent 调度、工具调用和结果校验模块，提升系统可解释性。"
        )

    def _generate_rewrite_tips(
        self,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> list[str]:
        tips = []

        if matched_skills:
            tips.append(
                f"简历中已体现 {', '.join(matched_skills[:5])}，建议在项目经历中突出这些技能的具体应用。"
            )

        if missing_skills:
            tips.append(
                f"岗位中还要求 {', '.join(missing_skills[:5])}，建议学习后补充到项目功能或技术栈中。"
            )

        tips.append("项目描述建议采用“技术栈 + 业务场景 + 具体功能 + 结果收益”的结构。")
        tips.append("不要直接写“精通”或“熟练掌握”未实际使用过的技术，避免简历内容夸大。")

        return tips