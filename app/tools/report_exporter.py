from datetime import datetime


def generate_markdown_report(result: dict) -> str:
    """根据多 Agent 分析结果生成 Markdown 报告"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    workflow = " → ".join(result.get("workflow", []))

    agent_summaries = result.get("agent_summaries", [])
    matched_skills = result.get("matched_skills", [])
    missing_skills = result.get("missing_skills", [])
    resume_skills = result.get("resume_skills", [])
    jd_skills = result.get("jd_skills", [])
    suggestions = result.get("suggestions", [])
    rewrite_tips = result.get("rewrite_tips", [])
    risk_items = result.get("risk_items", [])
    safe_suggestions = result.get("safe_suggestions", [])
    interview_questions = result.get("interview_questions", [])
    answer_tips = result.get("answer_tips", [])

    report = f"""# 多 Agent 求职辅助分析报告

生成时间：{now}

---

## 一、Agent 工作流

{workflow}

---

## 二、Agent 执行摘要

"""

    for idx, summary in enumerate(agent_summaries, start=1):
        report += f"{idx}. {summary}\n"

    report += f"""

---

## 三、岗位匹配度

岗位匹配度：**{result.get("match_score", 0)}%**

---

## 四、技能匹配分析

### 1. 简历已有技能

{format_list(resume_skills)}

### 2. 岗位要求技能

{format_list(jd_skills)}

### 3. 匹配技能

{format_list(matched_skills)}

### 4. 缺失技能

{format_list(missing_skills)}

---

## 五、优化建议

{format_numbered_list(suggestions)}

---

## 六、项目经历优化版本

{result.get("rewritten_project", "暂无项目经历优化内容。")}

---

## 七、改写注意事项

{format_numbered_list(rewrite_tips)}

---

## 八、ReviewAgent 校验结果

校验状态：**{result.get("review_status", "暂无")}**

### 风险提示

{format_list(risk_items) if risk_items else "未发现明显风险。"}

### 安全改写建议

{format_numbered_list(safe_suggestions)}

---

## 九、InterviewAgent 面试题

{format_numbered_list(interview_questions)}

---

## 十、面试回答建议

{format_numbered_list(answer_tips)}

---

## 十一、说明

本报告由多 Agent 求职辅助系统自动生成，结果仅作为简历优化和面试准备参考。简历内容应以真实经历为准，避免虚构和过度夸大。
"""

    return report


def format_list(items: list[str]) -> str:
    if not items:
        return "暂无"

    return "\n".join([f"- {item}" for item in items])


def format_numbered_list(items: list[str]) -> str:
    if not items:
        return "暂无"

    return "\n".join([f"{idx}. {item}" for idx, item in enumerate(items, start=1)])