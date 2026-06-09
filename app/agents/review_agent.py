class ReviewAgent:
    """结果校验 Agent：检查改写结果是否存在夸大、虚构或技能冲突。"""

    RISK_WORDS = [
        "精通",
        "主导",
        "独立负责",
        "生产环境",
        "大规模",
        "高并发",
        "千万级",
        "百万级",
        "企业级",
        "完整落地",
        "显著提升",
        "熟练掌握",
        "深入掌握",
    ]

    def run(
        self,
        resume_text: str,
        jd_text: str,
        rewritten_project: str,
        resume_skills: list[str],
        jd_skills: list[str],
    ) -> dict:
        try:
            risk_items: list[str] = []
            safe_suggestions: list[str] = []

            risk_items.extend(
                self._check_risk_words(rewritten_project)
            )

            risk_items.extend(
                self._check_skill_conflict(
                    rewritten_project=rewritten_project,
                    resume_skills=resume_skills,
                    jd_skills=jd_skills,
                )
            )

            if risk_items:
                review_status = "WARNING"

                safe_suggestions.append(
                    "建议将过强表述改为更稳妥、可验证的项目经历表达。"
                )
                safe_suggestions.append(
                    "原简历未体现的技能，应使用“了解、可扩展、为后续接入提供基础”等表达。"
                )
            else:
                review_status = "PASS"

                safe_suggestions.append(
                    "改写内容整体较安全，未发现明显虚构或过度夸大的表达。"
                )

            safe_suggestions.append(
                "简历描述建议遵循：真实经历 + 技术栈 + 具体任务 + 可验证结果。"
            )

            return {
                "agent_name": "ReviewAgent",
                "review_status": review_status,
                "risk_items": risk_items,
                "safe_suggestions": safe_suggestions,
                "summary": f"已完成改写内容校验，状态：{review_status}。",
            }

        except Exception as exc:
            # 即使校验过程异常，也必须返回完整字段，避免 LangGraph 中断
            return {
                "agent_name": "ReviewAgent",
                "review_status": "ERROR",
                "risk_items": [
                    f"ReviewAgent 校验异常：{exc}"
                ],
                "safe_suggestions": [
                    "当前校验模块发生异常，请人工检查改写内容。"
                ],
                "summary": "ReviewAgent 执行异常，已返回兜底结果。",
            }

    def _check_risk_words(
        self,
        rewritten_project: str,
    ) -> list[str]:
        risk_items = []

        if not rewritten_project:
            return ["改写结果为空，无法完成内容校验。"]

        for word in self.RISK_WORDS:
            if word in rewritten_project:
                risk_items.append(
                    f"检测到可能夸大的表达：{word}"
                )

        return risk_items

    def _check_skill_conflict(
        self,
        rewritten_project: str,
        resume_skills: list[str],
        jd_skills: list[str],
    ) -> list[str]:
        risk_items = []

        resume_skill_set = {
            skill.lower()
            for skill in resume_skills
            if skill
        }

        for skill in jd_skills:
            if not skill:
                continue

            skill_in_output = (
                skill.lower() in rewritten_project.lower()
            )

            skill_in_resume = (
                skill.lower() in resume_skill_set
            )

            if skill_in_output and not skill_in_resume:
                risk_items.append(
                    f"改写内容中出现原简历未体现的技能：{skill}"
                )

        return risk_items