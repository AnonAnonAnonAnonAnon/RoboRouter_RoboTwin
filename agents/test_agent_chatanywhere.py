import os, sys
from openai import AsyncOpenAI
from agents import (
    Agent, Runner,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# 关闭 OpenAI 官方 Tracing，避免第三方 key 报 401
set_tracing_disabled(True)

# 第三方通常只兼容 Chat Completions，切换掉默认的 Responses
set_default_openai_api("chat_completions")

# 用 ChatAnyWhere 的 OpenAI 兼容端点与密钥
base_url = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
api_key  = os.getenv("OPENAI_API_KEY")  # 必须已设置

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

client = AsyncOpenAI(base_url=base_url, api_key=api_key)
set_default_openai_client(client)

# 选一个在 /v1/models 里能看到的模型
model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

agent = Agent(name="Assistant", instructions="You are a helpful assistant.", model=model)

# 支持从命令行传自定义提示；未传就用默认示例
prompt = "Write a haiku about recursion in programming." if len(sys.argv) == 1 else " ".join(sys.argv[1:])
res = Runner.run_sync(agent, prompt)
print(res.final_output)



