import os
import uuid
from typing import Literal

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from app.observability.monitor import metrics_store
from app.agents.graph_workflow import run_graph_workflow
from app.rag.document_ingestion import ingest_document
from app.tools.file_loader import load_resume_file
from app.tools.report_exporter import generate_markdown_report


app = FastAPI(title="Job Agent LoRA API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_OPERATIONS = {"match", "rewrite", "interview", "full"}


class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str
    operation: Literal["match", "rewrite", "interview", "full"] = "full"


class ReportRequest(BaseModel):
    result: dict


@app.get("/observability/summary")
def observability_summary():
    return metrics_store.get_summary()


@app.get("/observability/recent")
def observability_recent(
    limit: int = 30,
):
    return {
        "runs": metrics_store.get_recent_runs(
            limit=limit
        )
    }

@app.get("/")
def root():
    return {"message": "Job Agent LoRA API is running."}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    return run_graph_workflow(
        resume_text=request.resume_text,
        jd_text=request.jd_text,
        operation=request.operation,
    )


@app.post("/analyze_file")
async def analyze_file(
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...),
    operation: str = Form("full"),
):
    if operation not in ALLOWED_OPERATIONS:
        operation = "full"

    original_filename = resume_file.filename or "resume.txt"
    file_ext = os.path.splitext(original_filename)[-1].lower()

    if file_ext not in {".txt", ".pdf", ".docx"}:
        return {
            "status": "error",
            "message": "仅支持 txt、pdf 和 docx 文件。",
        }

    saved_filename = f"{uuid.uuid4().hex}{file_ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    try:
        content = await resume_file.read()

        with open(saved_path, "wb") as file:
            file.write(content)

        resume_text = load_resume_file(saved_path)

        result = run_graph_workflow(
            resume_text=resume_text,
            jd_text=jd_text,
            operation=operation,
        )

        result["uploaded_filename"] = original_filename
        result["resume_text_preview"] = resume_text[:300]
        return result

    finally:
        if os.path.exists(saved_path):
            os.remove(saved_path)


@app.post("/export_report")
def export_report(request: ReportRequest):
    return {
        "report": generate_markdown_report(request.result)
    }


@app.post("/knowledge/upload")
async def upload_knowledge_document(
    file: UploadFile = File(...),
    category: str = Form("uploaded_document"),
    force_update: bool = Form(False),
):
    allowed_extensions = {".txt", ".pdf", ".docx"}

    original_filename = file.filename or "unknown_file"
    file_ext = os.path.splitext(original_filename)[1].lower()

    if file_ext not in allowed_extensions:
        return {
            "status": "error",
            "message": "仅支持 txt、pdf 和 docx 文件。",
        }

    saved_filename = f"{uuid.uuid4().hex}{file_ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    try:
        content = await file.read()

        with open(saved_path, "wb") as output_file:
            output_file.write(content)

        document_text = load_resume_file(saved_path)

        return ingest_document(
            document_text=document_text,
            filename=original_filename,
            category=category,
            force_update=force_update,
        )

    finally:
        if os.path.exists(saved_path):
            os.remove(saved_path)
