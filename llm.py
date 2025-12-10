from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
from time import sleep

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English"):
        if api_key:
            # --- 修复 1: 添加浏览器伪装头 ---
            # 这是一个标准的 Chrome User-Agent，用于欺骗 WAF 防火墙
            fake_browser_headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/"
            }
            
            # 初始化 OpenAI 客户端时注入 default_headers
            self.llm = OpenAI(
                api_key=api_key, 
                base_url=base_url,
                default_headers=fake_browser_headers
            )
            # self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            max_retries = 3

            # --- 核心修改开始 ---
            # 1. 准备你的 Packycode 自定义参数
            # 将 reasoning_effort, privacy, network 等所有非标参数都放入 extra_body
            custom_body = {
                "reasoning_effort": "high",  # 这里填你配置中的 high
                "disable_response_storage": True,
                "network_access": "enabled",
                "model_verbosity": "high"
            }

            request_kwargs = {
                "messages": messages,
                "model": self.model,
                "extra_body": custom_body, # 关键：通过 extra_body 透传参数
            }

            # 2. 判断是否为推理模型（防止 temperature 报错）
            # 如果是 gpt-5, o1, o3, codex-max 等推理模型，通常不能传 temperature
            is_reasoning_model = any(k in self.model.lower() for k in ["gpt-5", "o1-", "o3-", "reasoning", "codex-max"])

            if not is_reasoning_model:
                # 只有非推理模型才加 temperature=0
                request_kwargs["temperature"] = 0
            # --- 修改结束 ---
            
            for attempt in range(max_retries):
                try:
                    # 使用解包参数调用
                    response = self.llm.chat.completions.create(**request_kwargs)
                    # response = self.llm.chat.completions.create(messages=messages, temperature=0, model=self.model)
                    break
                except Exception as e:
                    # 打印更详细的错误信息，包括响应体（如果有）
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(5) # 延长重试等待时间，避免触发频率限制
                    # logger.error(f"Attempt {attempt + 1} failed: {e}")
                    # if attempt == max_retries - 1:
                    #     raise
                    # sleep(3)
            return response.choices[0].message.content
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            return response["choices"][0]["message"]["content"]

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM
