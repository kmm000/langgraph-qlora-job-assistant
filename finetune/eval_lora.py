import json
import re
from pathlib import Path

from finetune.infer_lora import load_model, generate_resume_text


TEST_FILE = "finetune/data/test.jsonl"
OUTPUT_FILE = "finetune/output/lora_eval_results.json"


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


SKILL_KEYWORDS = [
    "Python",
    "FastAPI",
    "Streamlit",
    "LangGraph",
    "LangChain",
    "RAG",
    "Agent",
    "多Agent",
    "工具调用",
    "Prompt",
    "LoRA",
    "模型微调",
    "PyTorch",
    "YOLOv5",
    "Embedding",
    "向量数据库",
    "FAISS",
    "Chroma",
    "Docker",
    "Linux",
    "Pandas",
    "Matplotlib",
]


def load_test_samples(file_path: str) -> list[dict]:
    samples = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)
            messages = item["messages"]

            user_content = messages[1]["content"]
            reference_output = messages[2]["content"]

            target_job = extract_field(user_content, "目标岗位")
            job_requirements = extract_field(user_content, "岗位要求")
            original_project = extract_field(user_content, "原始项目")

            samples.append(
                {
                    "target_job": target_job,
                    "job_requirements": job_requirements,
                    "original_project": original_project,
                    "reference_output": reference_output,
                }
            )

    return samples


def extract_field(text: str, field_name: str) -> str:
    pattern = rf"【{field_name}】(.*?)(?=【|$)"
    match = re.search(pattern, text, re.S)

    if match:
        return match.group(1).strip()

    return ""


def count_bullets(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("- ")]
    return len(bullet_lines)


def check_format(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return False

    return all(line.startswith("- ") for line in lines)


def check_bullet_count(text: str) -> bool:
    bullet_count = count_bullets(text)
    return 3 <= bullet_count <= 5


def keyword_coverage(text: str, job_requirements: str) -> dict:
    required_keywords = [
        keyword for keyword in SKILL_KEYWORDS
        if keyword.lower() in job_requirements.lower()
    ]

    if not required_keywords:
        return {
            "required_keywords": [],
            "covered_keywords": [],
            "coverage": 0.0,
        }

    covered_keywords = [
        keyword for keyword in required_keywords
        if keyword.lower() in text.lower()
    ]

    coverage = round(len(covered_keywords) / len(required_keywords) * 100, 2)

    return {
        "required_keywords": required_keywords,
        "covered_keywords": covered_keywords,
        "coverage": coverage,
    }


def count_risk_words(text: str) -> dict:
    hit_words = [word for word in RISK_WORDS if word in text]

    return {
        "risk_count": len(hit_words),
        "risk_words": hit_words,
    }


def evaluate_one(generated_text: str, job_requirements: str) -> dict:
    bullet_count = count_bullets(generated_text)
    format_ok = check_format(generated_text)
    bullet_count_ok = check_bullet_count(generated_text)
    keyword_result = keyword_coverage(generated_text, job_requirements)
    risk_result = count_risk_words(generated_text)

    return {
        "format_ok": format_ok,
        "bullet_count": bullet_count,
        "bullet_count_ok": bullet_count_ok,
        "keyword_coverage": keyword_result["coverage"],
        "required_keywords": keyword_result["required_keywords"],
        "covered_keywords": keyword_result["covered_keywords"],
        "risk_count": risk_result["risk_count"],
        "risk_words": risk_result["risk_words"],
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)

    if total == 0:
        return {}

    format_ok_count = sum(1 for item in results if item["metrics"]["format_ok"])
    bullet_count_ok_count = sum(1 for item in results if item["metrics"]["bullet_count_ok"])
    avg_keyword_coverage = sum(item["metrics"]["keyword_coverage"] for item in results) / total
    avg_risk_count = sum(item["metrics"]["risk_count"] for item in results) / total

    return {
        "total_samples": total,
        "format_follow_rate": round(format_ok_count / total * 100, 2),
        "bullet_count_pass_rate": round(bullet_count_ok_count / total * 100, 2),
        "avg_keyword_coverage": round(avg_keyword_coverage, 2),
        "avg_risk_count": round(avg_risk_count, 2),
    }


def main():
    Path("finetune/output").mkdir(parents=True, exist_ok=True)

    samples = load_test_samples(TEST_FILE)

    print(f"测试集样本数：{len(samples)}")
    print("正在加载 LoRA 模型……")

    tokenizer, model = load_model()

    results = []

    for idx, sample in enumerate(samples, start=1):
        print(f"正在评估第 {idx}/{len(samples)} 条……")

        generated_text = generate_resume_text(
            tokenizer=tokenizer,
            model=model,
            target_job=sample["target_job"],
            job_requirements=sample["job_requirements"],
            original_project=sample["original_project"],
        )

        metrics = evaluate_one(
            generated_text=generated_text,
            job_requirements=sample["job_requirements"],
        )

        results.append(
            {
                "id": idx,
                "target_job": sample["target_job"],
                "job_requirements": sample["job_requirements"],
                "original_project": sample["original_project"],
                "reference_output": sample["reference_output"],
                "generated_output": generated_text,
                "metrics": metrics,
            }
        )

    summary = summarize(results)

    output = {
        "summary": summary,
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("LoRA 评估完成")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n详细结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()