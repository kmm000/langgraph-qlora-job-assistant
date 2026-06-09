import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "finetune/output/resume_qlora_adapter"

SYSTEM_PROMPT = (
    "你是一名专业的中文技术简历优化助手。"
    "只能基于候选人的真实经历进行改写，不得虚构技术、成果或职责。"
    "输出3至5条简历项目描述，每条以“- ”开头。"
)


class LoraRewriteClient:
    """LoRA 简历改写模型客户端：第一次调用时加载模型，后续复用。"""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.loaded = False

    def load_model(self):
        if self.loaded:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("当前环境无法使用 CUDA，无法加载 LoRA 模型。")

        print("正在加载 LoRA 简历改写模型……")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=False,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=False,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            is_trainable=False,
        )

        self.model.eval()
        self.loaded = True

        print("LoRA 简历改写模型加载完成。")

    def rewrite(
            self,
            resume_text: str,
            jd_text: str,
            matched_skills: list[str],
            missing_skills: list[str],
            rag_contexts: list[str] | None = None,
    ) -> str:
        self.load_model()

        rag_context_text = "\n\n".join(
            rag_contexts or []
        )

        user_prompt = (
            f"【岗位 JD】\n{jd_text}\n\n"
            f"【候选人原始简历】\n{resume_text}\n\n"
            f"【已匹配技能】\n{matched_skills}\n\n"
            f"【缺失技能】\n{missing_skills}\n\n"
            f"【检索到的简历优化知识】\n"
            f"{rag_context_text}\n\n"
            "请结合检索知识优化项目经历。"
            "检索知识只能作为表达方式参考，"
            "不得将候选人未实际使用的技术写成真实经历。"
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

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=240,
                do_sample=True,
                temperature=0.25,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        result = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return result.strip()


# 全局单例：保证模型只加载一次
lora_rewrite_client = LoraRewriteClient()