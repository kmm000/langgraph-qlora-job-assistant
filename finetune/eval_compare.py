import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "finetune/output/resume_qlora_adapter"

TEST_FILE = "finetune/data/test.jsonl"
OUTPUT_FILE = "finetune/output/base_vs_lora_eval_results.json"

SYSTEM_PROMPT = (
    "你是一名专业的中文技术简历优化助手。"
    "只能基于候选人的真实经历进行改写，不得虚构技术、成果或职责。"
    "输出3至5条简历项目描述，每条以“- ”开头。"
)

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

            samples.append(
                {
                    "target_job": extract_field(user_content, "目标岗位"),
                    "job_requirements": extract_field(user_content, "岗位要求"),
                    "original_project": extract_field(user_content, "原始项目"),
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


def load_base_model():
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA。")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.eval()

    return tokenizer, model


def load_lora_model():
    tokenizer, base_model = load_base_model()

    lora_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    lora_model.eval()

    return tokenizer, lora_model


def generate_text(
    tokenizer,
    model,
    target_job: str,
    job_requirements: str,
    original_project: str,
) -> str:
    user_prompt = (
        f"【目标岗位】{target_job}\n"
        f"【岗位要求】{job_requirements}\n"
        f"【原始项目】{original_project}"
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=320,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return result.strip()


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


def summarize(results: list[dict], model_key: str) -> dict:
    total = len(results)

    if total == 0:
        return {}

    format_ok_count = sum(1 for item in results if item[model_key]["metrics"]["format_ok"])
    bullet_count_ok_count = sum(1 for item in results if item[model_key]["metrics"]["bullet_count_ok"])
    avg_keyword_coverage = sum(item[model_key]["metrics"]["keyword_coverage"] for item in results) / total
    avg_risk_count = sum(item[model_key]["metrics"]["risk_count"] for item in results) / total

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

    print("\n正在加载基础模型……")
    base_tokenizer, base_model = load_base_model()

    base_results = []

    for idx, sample in enumerate(samples, start=1):
        print(f"基础模型评估：{idx}/{len(samples)}")

        base_output = generate_text(
            tokenizer=base_tokenizer,
            model=base_model,
            target_job=sample["target_job"],
            job_requirements=sample["job_requirements"],
            original_project=sample["original_project"],
        )

        base_results.append(
            {
                "sample": sample,
                "base_output": base_output,
                "base_metrics": evaluate_one(base_output, sample["job_requirements"]),
            }
        )

    del base_model
    torch.cuda.empty_cache()

    print("\n正在加载 LoRA 模型……")
    lora_tokenizer, lora_model = load_lora_model()

    results = []

    for idx, item in enumerate(base_results, start=1):
        sample = item["sample"]

        print(f"LoRA模型评估：{idx}/{len(base_results)}")

        lora_output = generate_text(
            tokenizer=lora_tokenizer,
            model=lora_model,
            target_job=sample["target_job"],
            job_requirements=sample["job_requirements"],
            original_project=sample["original_project"],
        )

        results.append(
            {
                "id": idx,
                "target_job": sample["target_job"],
                "job_requirements": sample["job_requirements"],
                "original_project": sample["original_project"],
                "reference_output": sample["reference_output"],
                "base": {
                    "output": item["base_output"],
                    "metrics": item["base_metrics"],
                },
                "lora": {
                    "output": lora_output,
                    "metrics": evaluate_one(lora_output, sample["job_requirements"]),
                },
            }
        )

    summary = {
        "base_model": summarize(results, "base"),
        "lora_model": summarize(results, "lora"),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("基础模型 vs LoRA 模型评估完成")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n详细结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()