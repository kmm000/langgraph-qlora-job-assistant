import requests


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    """调用本地 Ollama 模型生成文本"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return "【LLM调用失败】无法连接 Ollama 服务，请确认 Ollama 已启动。"

    except requests.exceptions.Timeout:
        return "【LLM调用失败】Ollama 响应超时，请换用更小模型或稍后重试。"

    except Exception as e:
        return f"【LLM调用失败】{e}"