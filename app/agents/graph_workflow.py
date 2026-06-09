from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.interview_agent import InterviewAgent
from app.agents.jd_agent import JDAgent
from app.agents.match_agent import MatchAgent
from app.agents.rag_agent import RAGAgent
from app.agents.resume_agent import ResumeAgent
from app.agents.review_agent import ReviewAgent
from app.agents.rewrite_agent import RewriteAgent
from time import perf_counter

from app.observability.monitor import (
    metrics_store,
    new_request_id,
    observe_node,
)

class JobAgentState(TypedDict, total=False):
    resume_text: str
    jd_text: str
    operation: str
    request_id: str
    node_traces: List[Dict[str, Any]]
    total_latency_ms: float
    skill_score: float
    semantic_score: float
    project_score: float
    project_scores: List[Dict[str, Any]]
    score_weights: Dict[str, float]
    embedding_available: bool

    workflow: List[str]
    agent_summaries: List[str]

    resume_skills: List[str]
    jd_skills: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    suggestions: List[str]
    match_score: float

    rag_contexts: List[str]
    rag_results: List[Dict[str, Any]]

    rewritten_project: str
    rewrite_tips: List[str]

    review_status: str
    risk_items: List[str]
    safe_suggestions: List[str]

    interview_questions: List[str]
    answer_tips: List[str]


resume_agent = ResumeAgent()
jd_agent = JDAgent()
match_agent = MatchAgent()
rag_agent = RAGAgent()
rewrite_agent = RewriteAgent()
review_agent = ReviewAgent()
interview_agent = InterviewAgent()

@observe_node("ResumeAgent")
def resume_node(state: JobAgentState) -> JobAgentState:
    result = resume_agent.run(state["resume_text"])

    return {
        **state,
        "workflow": state.get("workflow", []) + [result.get("agent_name", "ResumeAgent")],
        "agent_summaries": state.get("agent_summaries", []) + [result.get("summary", "")],
        "resume_skills": result.get("resume_skills", []),
    }

@observe_node("JDAgent")
def jd_node(state: JobAgentState) -> JobAgentState:
    result = jd_agent.run(state["jd_text"])

    return {
        **state,
        "workflow": state.get("workflow", []) + [result.get("agent_name", "JDAgent")],
        "agent_summaries": state.get("agent_summaries", []) + [result.get("summary", "")],
        "jd_skills": result.get("jd_skills", []),
    }

@observe_node("MatchAgent")
def match_node(
    state: JobAgentState,
) -> JobAgentState:
    result = match_agent.run(
        resume_text=state["resume_text"],
        jd_text=state["jd_text"],
        resume_skills=state.get(
            "resume_skills",
            [],
        ),
        jd_skills=state.get(
            "jd_skills",
            [],
        ),
    )

    return {
        **state,
        "workflow": state.get(
            "workflow",
            [],
        ) + [
            result.get(
                "agent_name",
                "MatchAgent",
            )
        ],
        "agent_summaries": state.get(
            "agent_summaries",
            [],
        ) + [
            result.get(
                "summary",
                "",
            )
        ],
        "match_score": result.get(
            "match_score",
            0.0,
        ),
        "skill_score": result.get(
            "skill_score",
            0.0,
        ),
        "semantic_score": result.get(
            "semantic_score",
            0.0,
        ),
        "project_score": result.get(
            "project_score",
            0.0,
        ),
        "resume_skills": result.get(
            "resume_skills",
            [],
        ),
        "jd_skills": result.get(
            "jd_skills",
            [],
        ),
        "matched_skills": result.get(
            "matched_skills",
            [],
        ),
        "missing_skills": result.get(
            "missing_skills",
            [],
        ),
        "suggestions": result.get(
            "suggestions",
            [],
        ),
        "project_scores": result.get(
            "project_scores",
            [],
        ),
        "score_weights": result.get(
            "score_weights",
            {},
        ),
        "embedding_available": result.get(
            "embedding_available",
            False,
        ),
    }

@observe_node("RAGAgent")
def rag_node(state: JobAgentState) -> JobAgentState:
    result = rag_agent.run(
        resume_text=state["resume_text"],
        jd_text=state["jd_text"],
        top_k=4,
    )

    return {
        **state,
        "workflow": state.get("workflow", []) + [result.get("agent_name", "RAGAgent")],
        "agent_summaries": state.get("agent_summaries", []) + [result.get("summary", "")],
        "rag_contexts": result.get("rag_contexts", []),
        "rag_results": result.get("rag_results", []),
    }

@observe_node("RewriteAgent")
def rewrite_node(state: JobAgentState) -> JobAgentState:
    result = rewrite_agent.run(
        resume_text=state["resume_text"],
        jd_text=state["jd_text"],
        matched_skills=state.get("matched_skills", []),
        missing_skills=state.get("missing_skills", []),
        rag_contexts=state.get("rag_contexts", []),
    )

    return {
        **state,
        "workflow": state.get("workflow", []) + [result.get("agent_name", "RewriteAgent")],
        "agent_summaries": state.get("agent_summaries", []) + [result.get("summary", "")],
        "rewritten_project": result.get("rewritten_project", ""),
        "rewrite_tips": result.get("rewrite_tips", []),
    }

@observe_node("ReviewAgent")
def review_node(state: JobAgentState) -> JobAgentState:
    result = review_agent.run(
        resume_text=state["resume_text"],
        jd_text=state["jd_text"],
        rewritten_project=state.get("rewritten_project", ""),
        resume_skills=state.get("resume_skills", []),
        jd_skills=state.get("jd_skills", []),
    )

    return {
        **state,
        "workflow": state.get("workflow", []) + [result.get("agent_name", "ReviewAgent")],
        "agent_summaries": state.get("agent_summaries", []) + [
            result.get("summary", "ReviewAgent 未返回执行摘要。")
        ],
        "review_status": result.get("review_status", "ERROR"),
        "risk_items": result.get("risk_items", []),
        "safe_suggestions": result.get("safe_suggestions", []),
    }

@observe_node("InterviewAgent")
def interview_node(state: JobAgentState) -> JobAgentState:
    result = interview_agent.run(
        resume_skills=state.get("resume_skills", []),
        jd_skills=state.get("jd_skills", []),
        matched_skills=state.get("matched_skills", []),
        missing_skills=state.get("missing_skills", []),
        match_score=state.get("match_score", 0.0),
    )

    return {
        **state,
        "workflow": state.get("workflow", []) + [result.get("agent_name", "InterviewAgent")],
        "agent_summaries": state.get("agent_summaries", []) + [result.get("summary", "")],
        "interview_questions": result.get("interview_questions", []),
        "answer_tips": result.get("answer_tips", []),
    }


def route_after_match(state: JobAgentState) -> str:
    operation = state.get("operation", "full")
    return operation if operation in {"match", "rewrite", "interview", "full"} else "full"


def route_after_rag(state: JobAgentState) -> str:
    return "interview" if state.get("operation") == "interview" else "rewrite"


def route_after_review(state: JobAgentState) -> str:
    return "interview" if state.get("operation") == "full" else "end"


def build_job_agent_graph():
    graph = StateGraph(JobAgentState)

    graph.add_node("resume_node", resume_node)
    graph.add_node("jd_node", jd_node)
    graph.add_node("match_node", match_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("rewrite_node", rewrite_node)
    graph.add_node("review_node", review_node)
    graph.add_node("interview_node", interview_node)

    graph.set_entry_point("resume_node")

    graph.add_edge("resume_node", "jd_node")
    graph.add_edge("jd_node", "match_node")

    graph.add_conditional_edges(
        "match_node",
        route_after_match,
        {
            "match": END,
            "rewrite": "rag_node",
            "interview": "rag_node",
            "full": "rag_node",
        },
    )

    graph.add_conditional_edges(
        "rag_node",
        route_after_rag,
        {
            "rewrite": "rewrite_node",
            "interview": "interview_node",
        },
    )

    graph.add_edge("rewrite_node", "review_node")

    graph.add_conditional_edges(
        "review_node",
        route_after_review,
        {
            "interview": "interview_node",
            "end": END,
        },
    )

    graph.add_edge("interview_node", END)

    return graph.compile()


job_agent_graph = build_job_agent_graph()


def run_graph_workflow(
    resume_text: str,
    jd_text: str,
    operation: str = "full",
) -> Dict[str, Any]:
    request_id = new_request_id()

    init_state: JobAgentState = {
        "request_id": request_id,
        "resume_text": resume_text,
        "jd_text": jd_text,
        "operation": operation,
        "workflow": [],
        "agent_summaries": [],
        "node_traces": [],
    }

    start_time = perf_counter()

    try:
        result = job_agent_graph.invoke(
            init_state
        )

        total_latency_ms = (
            perf_counter() - start_time
        ) * 1000

        metrics_store.record_request(
            request_id=request_id,
            operation=operation,
            status="success",
            total_latency_ms=total_latency_ms,
        )

        return {
            **result,
            "request_id": request_id,
            "total_latency_ms": round(
                total_latency_ms,
                2,
            ),
        }

    except Exception as exc:
        total_latency_ms = (
            perf_counter() - start_time
        ) * 1000

        metrics_store.record_request(
            request_id=request_id,
            operation=operation,
            status="failed",
            total_latency_ms=total_latency_ms,
            error_message=str(exc)[:1000],
        )

        raise
