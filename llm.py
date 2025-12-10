from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
from time import sleep

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English"):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
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
            # --- 修改开始：自定义参数配置 ---
            # 1. 定义你的高级参数
            # 注意：OpenAI官方SDK支持 reasoning_effort 参数，但其他非标参数建议放入 extra_body
            request_kwargs = {
                "messages": messages,
                "model": self.model,
            }
            # 2. 判断是否为推理模型（根据你的模型名特征）
            # 如果模型名包含 gpt-5, o1, o3 或 codex-max，通常被视为推理模型
            is_reasoning_model = any(k in self.model.lower() for k in ["gpt-5", "o1-", "o3-", "reasoning", "codex-max"])
            if is_reasoning_model:
                # 推理模型通常不支持 temperature，或者需要设为 1 (视具体 provider 而定)
                # 这里我们选择不传 temperature，让 API 使用默认值
                request_kwargs["reasoning_effort"] = "high" # 或 "medium", "low". 注意：OpenAI SDK 这里的参数值通常是固定的
            else:
                # 普通模型保留原逻辑
                request_kwargs["temperature"] = 0

            # 3. 传递 packycode 特有的额外参数 (如下载响应存储、联网等)
            # 使用 extra_body 可以将参数直接透传给 API JSON body
            request_kwargs["extra_body"] = {
                "disable_response_storage": True,
                "network_access": "enabled",
                # 如果SDK不识别 reasoning_effort，也可以把 reasoning_effort 放这里
                # "reasoning_effort": "xhigh" 
            }
            # --- 修改结束 ---
            for attempt in range(max_retries):
                try:
                    # 使用解包参数调用
                    response = self.llm.chat.completions.create(**request_kwargs)
                    # response = self.llm.chat.completions.create(messages=messages, temperature=0, model=self.model)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(3)
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
