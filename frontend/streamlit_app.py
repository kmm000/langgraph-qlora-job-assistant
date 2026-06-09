import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


def show_result(result: dict):
    st.divider()
    st.subheader("分析结果")

    workflow = result.get("workflow", [])
    st.markdown("### Agent 工作流")
    if workflow:
        st.write(" → ".join(workflow))
    else:
        st.info("暂无 Agent 工作流信息。")
    request_id = result.get("request_id", "")
    total_latency_ms = result.get(
        "total_latency_ms",
        0,
    )
    node_traces = result.get(
        "node_traces",
        [],
    )

    with st.expander(
            "运行追踪与耗时",
            expanded=False,
    ):
        if request_id:
            st.code(
                f"Request ID: {request_id}"
            )

        st.metric(
            "工作流总耗时",
            f"{total_latency_ms / 1000:.2f} 秒",
        )

        if node_traces:
            trace_rows = []

            for trace in node_traces:
                trace_rows.append(
                    {
                        "Agent": trace.get(
                            "node_name",
                            "",
                        ),
                        "状态": trace.get(
                            "status",
                            "",
                        ),
                        "耗时（ms）": trace.get(
                            "latency_ms",
                            0,
                        ),
                        "错误": trace.get(
                            "error_message",
                            "",
                        )
                                or "",
                    }
                )

            st.dataframe(
                trace_rows,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("暂无节点追踪信息。")

    summaries = result.get("agent_summaries", [])
    st.markdown("### Agent 执行摘要")
    if summaries:
        for summary in summaries:
            st.info(summary)
    else:
        st.info("暂无 Agent 执行摘要。")

    st.markdown("### 岗位匹配度")

    score_col1, score_col2, score_col3, score_col4 = st.columns(4)

    with score_col1:
        st.metric(
            "综合匹配度",
            f"{result.get('match_score', 0)}%",
        )

    with score_col2:
        st.metric(
            "技能匹配",
            f"{result.get('skill_score', 0)}%",
            help="基于简历技能与岗位技能的精确匹配结果。",
        )

    with score_col3:
        st.metric(
            "语义相似度",
            f"{result.get('semantic_score', 0)}%",
            help="使用 bge-m3 计算完整简历与岗位 JD 的语义相似度。",
        )

    with score_col4:
        st.metric(
            "项目相关性",
            f"{result.get('project_score', 0)}%",
            help="计算简历项目段落与岗位 JD 的语义相关程度。",
        )

    project_scores = result.get(
        "project_scores",
        [],
    )

    if project_scores:
        st.markdown("### 项目相关性详情")

        for index, item in enumerate(
                project_scores,
                start=1,
        ):
            with st.expander(
                    f"项目片段 {index}："
                    f"{item.get('score', 0)}%"
            ):
                st.write(
                    item.get(
                        "project_preview",
                        "",
                    )
                )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 匹配技能")
        matched_skills = result.get("matched_skills", [])
        if matched_skills:
            for skill in matched_skills:
                st.success(skill)
        else:
            st.info("暂无匹配技能")

    with c2:
        st.markdown("### 缺失技能")
        missing_skills = result.get("missing_skills", [])
        if missing_skills:
            for skill in missing_skills:
                st.warning(skill)
        else:
            st.success("暂无明显缺失技能")

    resume_skills = result.get("resume_skills", [])
    jd_skills = result.get("jd_skills", [])

    st.markdown("### 简历已有技能")
    st.write("、".join(resume_skills) if resume_skills else "未识别到技能关键词。")

    st.markdown("### 岗位要求技能")
    st.write("、".join(jd_skills) if jd_skills else "未识别到岗位技能关键词。")

    suggestions = result.get("suggestions", [])
    st.markdown("### 优化建议")
    if suggestions:
        for idx, suggestion in enumerate(suggestions, start=1):
            st.write(f"{idx}. {suggestion}")
    else:
        st.info("暂无优化建议。")

    rag_results = result.get("rag_results", [])
    if rag_results:
        st.markdown("### RAG 检索结果")
        for index, item in enumerate(rag_results, start=1):
            metadata = item.get("metadata", {})
            with st.expander(f"{index}. {metadata.get('title', '检索结果')}"):
                st.write(item.get("document", ""))
                st.caption(
                    f"类别：{metadata.get('category', 'unknown')} | "
                    f"距离：{item.get('distance', '')}"
                )

    rewritten_project = result.get("rewritten_project", "")
    if rewritten_project:
        st.markdown("### 项目经历优化版本")
        st.write(rewritten_project)

    rewrite_tips = result.get("rewrite_tips", [])
    if rewrite_tips:
        st.markdown("### 改写注意事项")
        for idx, tip in enumerate(rewrite_tips, start=1):
            st.write(f"{idx}. {tip}")

    review_status = result.get("review_status")
    if review_status:
        st.markdown("### ReviewAgent 校验结果")
        if review_status == "PASS":
            st.success(f"校验状态：{review_status}")
        else:
            st.warning(f"校验状态：{review_status}")

        risk_items = result.get("risk_items", [])
        st.markdown("#### 风险提示")
        if risk_items:
            for item in risk_items:
                st.error(item)
        else:
            st.success("未发现明显风险。")

        safe_suggestions = result.get("safe_suggestions", [])
        if safe_suggestions:
            st.markdown("#### 安全改写建议")
            for idx, suggestion in enumerate(safe_suggestions, start=1):
                st.write(f"{idx}. {suggestion}")

    interview_questions = result.get("interview_questions", [])
    if interview_questions:
        st.markdown("### InterviewAgent 面试题")
        for idx, question in enumerate(interview_questions, start=1):
            st.write(f"{idx}. {question}")

    answer_tips = result.get("answer_tips", [])
    if answer_tips:
        st.markdown("### 面试回答建议")
        for idx, tip in enumerate(answer_tips, start=1):
            st.info(f"{idx}. {tip}")

    st.markdown("### 导出分析报告")
    try:
        response = requests.post(
            f"{API_BASE_URL}/export_report",
            json={"result": result},
            timeout=30,
        )
        if response.status_code == 200:
            report = response.json().get("report", "")
            st.download_button(
                label="下载 Markdown 分析报告",
                data=report,
                file_name="job_agent_analysis_report.md",
                mime="text/markdown",
            )
        else:
            st.warning("报告生成接口调用失败。")
    except Exception as exc:
        st.warning(f"报告生成失败：{exc}")


def analyze_by_text(resume_text: str, jd_text: str, operation: str):
    return requests.post(
        f"{API_BASE_URL}/analyze",
        json={
            "resume_text": resume_text,
            "jd_text": jd_text,
            "operation": operation,
        },
        timeout=300,
    )


def analyze_by_file(resume_file, jd_text: str, operation: str):
    return requests.post(
        f"{API_BASE_URL}/analyze_file",
        files={
            "resume_file": (
                resume_file.name,
                resume_file.getvalue(),
                resume_file.type,
            )
        },
        data={
            "jd_text": jd_text,
            "operation": operation,
        },
        timeout=300,
    )


def main():
    st.set_page_config(
        page_title="多 Agent 求职辅助系统",
        layout="wide",
    )

    with st.sidebar:
        st.header("知识库管理")

        knowledge_file = st.file_uploader(
            "上传知识库文档",
            type=["txt", "pdf", "docx"],
            key="knowledge_file",
        )

        knowledge_category = st.selectbox(
            "文档类别",
            ["resume_case", "interview", "job", "uploaded_document"],
        )

        force_update = st.checkbox(
            "已存在时重新导入",
            value=False,
        )

        if st.button("导入向量库"):
            if knowledge_file is None:
                st.warning("请先选择文档。")
            else:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/knowledge/upload",
                        files={
                            "file": (
                                knowledge_file.name,
                                knowledge_file.getvalue(),
                                knowledge_file.type,
                            )
                        },
                        data={
                            "category": knowledge_category,
                            "force_update": str(force_update).lower(),
                        },
                        timeout=300,
                    )

                    try:
                        result = response.json()
                    except ValueError:
                        st.error(f"后端返回的不是 JSON，状态码：{response.status_code}")
                        st.code(response.text)
                        result = None

                    if result is not None:
                        if response.status_code != 200:
                            st.error(result.get("message", "知识库文档导入失败。"))
                        elif result.get("status") == "already_exists":
                            st.info(
                                f"{result.get('message', '文档已存在')}\n\n"
                                f"文件：{result.get('filename', '')}\n\n"
                                f"SHA256：{result.get('document_hash', '')}\n\n"
                                f"已有分块：{result.get('chunk_count', 0)}"
                            )
                        elif result.get("status") == "inserted":
                            st.success(
                                f"{result.get('message', '文档导入成功')}\n\n"
                                f"文件：{result.get('filename', '')}\n\n"
                                f"SHA256：{result.get('document_hash', '')}\n\n"
                                f"写入分块：{result.get('chunk_count', 0)}"
                            )
                        else:
                            st.warning(result)

                except requests.exceptions.ConnectionError:
                    st.error("无法连接 FastAPI 后端，请确认后端已经启动。")
                except requests.exceptions.Timeout:
                    st.error("知识库导入超时，请稍后重试。")
                except Exception as exc:

                    st.error(f"知识库导入失败：{exc}")

            st.divider()
            st.subheader("运行监控")

            if st.button(
                    "刷新运行指标",
                    key="refresh_metrics",
            ):
                try:
                    response = requests.get(
                        f"{API_BASE_URL}/observability/summary",
                        timeout=15,
                    )

                    if response.status_code == 200:
                        metrics = response.json()
                        request_metrics = metrics.get(
                            "requests",
                            {},
                        )

                        st.metric(
                            "累计请求",
                            request_metrics.get(
                                "total",
                                0,
                            ),
                        )

                        st.metric(
                            "请求成功率",
                            f"{request_metrics.get('success_rate', 0)}%",
                        )

                        st.metric(
                            "平均响应时间",
                            f"{request_metrics.get('avg_latency_ms', 0) / 1000:.2f}s",
                        )

                        with st.expander(
                                "Agent 运行指标"
                        ):
                            st.dataframe(
                                metrics.get(
                                    "agents",
                                    [],
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                    else:
                        st.warning("运行指标读取失败。")

                except Exception as exc:
                    st.warning(
                        f"监控接口调用失败：{exc}"
                    )



    st.title("基于 LoRA 微调的多 Agent 求职辅助系统")
    st.caption("当前版本：LangGraph 多 Agent + QLoRA + ChromaDB RAG")

    st.markdown(
        "本系统支持简历解析、岗位 JD 分析、匹配度评估、"
        "项目经历改写、真实性校验、面试题生成和知识库检索。"
    )

    operation_label = st.radio(
        "请选择分析任务",
        ["仅匹配分析", "简历优化", "面试准备", "完整分析"],
        horizontal=True,
    )

    operation_map = {
        "仅匹配分析": "match",
        "简历优化": "rewrite",
        "面试准备": "interview",
        "完整分析": "full",
    }
    operation = operation_map[operation_label]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("简历内容")

        input_mode = st.radio(
            "请选择简历输入方式",
            ["粘贴文本", "上传文件"],
            horizontal=True,
            key="resume_input_mode",
        )

        resume_text = ""
        resume_file = None

        if input_mode == "粘贴文本":
            resume_text = st.text_area(
                "请粘贴简历文本",
                height=350,
                placeholder="请输入你的简历内容...",
            )
        else:
            resume_file = st.file_uploader(
                "请上传简历文件",
                type=["txt", "docx", "pdf"],
                key="resume_file",
            )
            if resume_file is not None:
                st.success(f"已选择文件：{resume_file.name}")

    with col2:
        st.subheader("岗位 JD")
        jd_text = st.text_area(
            "请粘贴岗位 JD",
            height=350,
            placeholder="请输入岗位描述...",
        )

    if st.button("开始分析", type="primary"):
        try:
            if input_mode == "粘贴文本":
                if not resume_text.strip() or not jd_text.strip():
                    st.warning("请同时输入简历内容和岗位 JD。")
                    return

                with st.spinner("正在分析..."):
                    response = analyze_by_text(
                        resume_text=resume_text,
                        jd_text=jd_text,
                        operation=operation,
                    )

            else:
                if resume_file is None or not jd_text.strip():
                    st.warning("请上传简历文件，并输入岗位 JD。")
                    return

                with st.spinner("正在读取简历文件并分析..."):
                    response = analyze_by_file(
                        resume_file=resume_file,
                        jd_text=jd_text,
                        operation=operation,
                    )

            if response.status_code != 200:
                st.error(f"接口请求失败，状态码：{response.status_code}")
                st.code(response.text)
                return

            result = response.json()

            if result.get("status") == "error":
                st.error(result.get("message", "分析失败。"))
                return

            if result.get("uploaded_filename"):
                st.success(f"已读取文件：{result['uploaded_filename']}")

            if result.get("resume_text_preview"):
                st.caption("简历文本预览：")
                st.write(result["resume_text_preview"])

            show_result(result)

        except requests.exceptions.ConnectionError:
            st.error("无法连接 FastAPI 后端，请确认后端已启动。")
        except requests.exceptions.Timeout:
            st.error("请求超时，请稍后重试。")
        except Exception as exc:
            st.error(f"程序出错：{exc}")


if __name__ == "__main__":
    main()
