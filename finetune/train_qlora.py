import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# Hugging Face 基础模型
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# 数据路径
TRAIN_FILE = "finetune/data/train.jsonl"
TEST_FILE = "finetune/data/test.jsonl"

# LoRA Adapter 输出目录
OUTPUT_DIR = "finetune/output/resume_qlora_adapter"


def check_environment() -> None:
    """检查训练环境。"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前 PyTorch 无法使用 CUDA，请检查 PyTorch CUDA 版本和显卡驱动。"
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("=" * 60)
    print(f"GPU：{gpu_name}")
    print(f"显存：{gpu_memory:.2f} GB")
    print(f"PyTorch：{torch.__version__}")
    print(f"CUDA：{torch.version.cuda}")
    print("=" * 60)


def main() -> None:
    check_environment()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # RTX 4060 支持 BF16，但 Windows/库版本兼容性不一致。
    # 第一版优先使用 FP16，更稳。
    compute_dtype = torch.bfloat16

    # 4bit QLoRA 量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print("正在加载 Tokenizer……")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("正在以 4bit 方式加载基础模型……")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # 关闭缓存，否则会影响梯度检查点
    model.config.use_cache = False

    # 准备量化模型进行 LoRA 训练
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    print("正在加载数据集……")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "test": TEST_FILE,
        },
    )

    print(f"训练集：{len(dataset['train'])} 条")
    print(f"测试集：{len(dataset['test'])} 条")

    # LoRA 配置
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # 训练参数：按 4060 8GB 设置
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # 数据与序列
        max_length=512,
        packing=False,

        # 训练轮次
        num_train_epochs=3,

        # 显存控制
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,

        # 优化器
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=0.3,

        # 精度
        fp16=False,
        bf16=True,

        # 日志与保存
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,

        # 其他
        report_to="none",
        seed=42,
        dataset_num_proc=1,
    )

    print("正在创建 SFTTrainer……")

    # 当前数据是 messages 对话格式。
    # 新版 SFTTrainer 可以自动识别对话数据并应用 chat template。
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("可训练参数：")
    trainer.model.print_trainable_parameters()

    print("开始训练……")
    trainer.train()

    print("正在保存 LoRA Adapter……")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 60)
    print("QLoRA 训练完成")
    print(f"Adapter 保存位置：{OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()