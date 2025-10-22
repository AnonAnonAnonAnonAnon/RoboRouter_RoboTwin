#成功运行

# /home/cyt/RoboRouter/RoboRouter_RoboTwin/agents/ca_agent_standalone.py
import sys
from openai import AsyncOpenAI
from agents import (
    Agent, Runner,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# ===== ① 在这里填你的网关与密钥（直接硬编码）=====
BASE_URL = "https://api.chatanywhere.tech/v1"   # 也可用 https://api.chatanywhere.org/v1
API_KEY  = "sk-AhGuNmK6xnFGdBCkFGpG0lcqj3TgLT7dQKU5JUSpaNQkUpZV"
MODEL    = "gpt-4o-mini"                        # 选 /v1/models 里存在的模型，如 gpt-3.5-turbo / gpt-4o 等
# ==============================================

# 关闭官方 Tracing（第三方 key 会 401）
set_tracing_disabled(True)

# 第三方一般只兼容 Chat Completions，切换掉默认 Responses
set_default_openai_api("chat_completions")

# 注入你自己的 OpenAI 兼容客户端
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client)

# 建一个最小 Agent
agent = Agent(name="Assistant", instructions="You are a helpful assistant.", model=MODEL)

if __name__ == "__main__":
    prompt = "Write a haiku about recursion in programming." if len(sys.argv) == 1 else " ".join(sys.argv[1:])
    res = Runner.run_sync(agent, prompt)
    print(res.final_output)
