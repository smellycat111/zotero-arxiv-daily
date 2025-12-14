import httpx
from loguru import logger
from time import sleep
import json

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.lang = lang
        
        # 如果没有设置 base_url，给一个默认值（防止本地运行报错）
        if not self.base_url:
            self.base_url = "https://www.right.codes/codex/v1/responses"

    def generate(self, messages: list[dict]) -> str:
        """
        使用自定义的 right.codes 格式发送请求
        """
        max_retries = 3
        
        # 1. 将 OpenAI 格式的 messages 转换为 right.codes 要求的多层嵌套格式
        # 原始格式: [{"role": "user", "content": "..."}]
        # 目标格式: "input": [{"type": "message", "role": "...", "content": [{"type": "input_text", "text": "..."}]}]
        custom_input = []
        for msg in messages:
            role = msg.get("role", "user")
            # 有些自定义接口不支持 "system" 角色，通常可以安全地转为 "user" 或 "assistant"
            # 这里我们尝试保留 system，如果报错再改
            if role == "system":
                # 将 system prompt 作为第一条 user 消息，或者视情况而定
                # 为了保险，很多非标接口建议把 system 改为 user
                role = "user" 
                
            content_text = msg.get("content", "")
            
            custom_msg = {
                "type": "message",
                "role": role,
                "content": [
                    {
                        "type": "input_text",
                        "text": content_text
                    }
                ]
            }
            custom_input.append(custom_msg)

        # 2. 构造请求 Payload
        payload = {
            "model": self.model,
            "input": custom_input,
            "stream": False,  # 我们需要一次性返回，不流式
            # 如果你的模型支持 reasoning_effort，可以在这里尝试添加，但看起来这个非标接口可能不支持
            # "reasoning_effort": "xhigh" 
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # 加上伪装头，以防万一
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }

        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to {self.base_url} with model {self.model}...")
                
                # 使用 httpx 发送原生 POST 请求
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        self.base_url, 
                        headers=headers, 
                        json=payload
                    )
                    
                # 检查状态码
                if response.status_code != 200:
                    logger.error(f"API Error {response.status_code}: {response.text}")
                    raise Exception(f"HTTP {response.status_code}")

                # 3. 解析响应
                # 由于不知道返回的具体 JSON 结构，我们需要尝试解析
                # 假设返回格式类似: {"output": "result..."} 或 standard OpenAI response
                resp_json = response.json()
                
                # 尝试打印一下响应结构以便调试（如果 Action 再次失败，查看日志里的这一行）
                # logger.debug(f"Raw Response: {resp_json}")

                # 这里需要根据实际返回做适配。
                # 既然 input 是自定义的，output 可能也是自定义的。
                # 常见可能性 1: 直接在顶层
                if "output" in resp_json:
                     # 有时候 output 是列表
                    if isinstance(resp_json["output"], list):
                         return resp_json["output"][0].get("text", "") or resp_json["output"][0].get("content", "")
                    return resp_json["output"]
                
                # 常见可能性 2: 模仿 OpenAI 结构
                if "choices" in resp_json:
                    return resp_json["choices"][0]["message"]["content"]
                
                # 常见可能性 3: 就在 content 里
                if "content" in resp_json:
                    return resp_json["content"]

                # 如果都找不到，直接返回整个 JSON 字符串让调用者处理（或者报错）
                logger.warning("Could not parse specific content field, returning full JSON dump.")
                return json.dumps(resp_json, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                sleep(5)
        return ""

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM
