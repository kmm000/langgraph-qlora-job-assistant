import json
from pathlib import Path

REQUIRED_ROLES = ["system", "user", "assistant"]

def validate_jsonl(file_path: str) -> None:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    valid_count = 0
    errors = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"第 {line_number} 行不是合法 JSON：{exc}")
                continue
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                errors.append(f"第 {line_number} 行 messages 必须包含3条消息")
                continue
            roles = [message.get("role") for message in messages]
            if roles != REQUIRED_ROLES:
                errors.append(f"第 {line_number} 行角色顺序错误，当前为 {roles}")
                continue
            contents = [message.get("content", "").strip() for message in messages]
            if not all(contents):
                errors.append(f"第 {line_number} 行存在空内容")
                continue
            assistant_output = contents[2]
            if not assistant_output.startswith("- "):
                errors.append(f"第 {line_number} 行 assistant 输出没有以 '- ' 开头")
                continue
            valid_count += 1
    print(f"有效数据：{valid_count} 条")
    if errors:
        print(f"发现问题：{len(errors)} 条")
        for error in errors:
            print(error)
        raise ValueError("数据校验失败，请修复以上问题。")
    print("数据格式校验通过。")

if __name__ == "__main__":
    validate_jsonl("finetune/data/train.jsonl")
    validate_jsonl("finetune/data/test.jsonl")
