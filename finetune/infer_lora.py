from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "finetune/output/resume_qlora_adapter"


SYSTEM_PROMPT = (
    "你是一名专业的中文技术简历优化助手。"
    "只能基于候选人的真实经历进行改写，不得虚构技术、成果或职责。"
    "输出3至5条简历项目描述，每条以“- ”开头。"
)


def check_paths() -> None:
    """检查 LoRA Adapter；基础模型既可以是 Hugging Face ID，也可以是本地目录。"""

    adapter_path = Path(ADAPTER_PATH)

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA Adapter 目录不存在：{adapter_path.resolve()}"
        )

    adapter_config = adapter_path / "adapter_config.json"
    adapter_weights = adapter_path / "adapter_model.safetensors"

    if not adapter_config.exists():
        raise FileNotFoundError(
            f"Adapter 目录中缺少：{adapter_config.resolve()}"
        )

    if not adapter_weights.exists():
        raise FileNotFoundError(
            f"Adapter 目录中缺少：{adapter_weights.resolve()}"
        )

    # 只有当 BASE_MODEL 是实际存在的本地路径时，才进行本地文件检查。
    base_model_path = Path(BASE_MODEL)

    if base_model_path.exists():
        config_path = base_model_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"本地基础模型目录缺少 config.json：{config_path.resolve()}"
            )

        print(f"使用本地基础模型：{base_model_path.resolve()}")
    else:
        print(f"使用 Hugging Face 模型 ID：{BASE_MODEL}")

    print(f"使用 LoRA Adapter：{adapter_path.resolve()}")


def load_model():
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

    print("正在加载基础模型……")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print("正在加载 LoRA Adapter……")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    model.eval()

    return tokenizer, model


def generate_resume_text(
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


def main() -> None:
    check_paths()

    tokenizer, model = load_model()

    target_job = "大模型应用开发实习生"

    job_requirements = (
        "熟悉 Python、FastAPI、LangGraph、多 Agent、"
        "工具调用、RAG 和模型微调。"
    )

    original_project = (
        "使用 FastAPI、Streamlit 和 LangGraph 开发多 Agent 求职辅助系统，"
        "支持简历解析、岗位 JD 分析、匹配度计算、项目经历改写、"
        "真实性校验和面试题生成。"
    )

    result = generate_resume_text(
        tokenizer=tokenizer,
        model=model,
        target_job=target_job,
        job_requirements=job_requirements,
        original_project=original_project,
    )

    print("\n" + "=" * 60)
    print("LoRA 模型输出")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()