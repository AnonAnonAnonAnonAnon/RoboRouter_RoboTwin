# doubao 测试,最小调用，自然语言问答

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === 配置区（按需修改） ===
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_ID = "doubao-seed-1-6-251015"   # 注意不要有换行或空格
API_KEY = "70f0d563-91a5-4704-a00e-f00cf3a9c864"  # 或留空用环境变量 ARK_API_KEY

# === 代码区（一般无需改动） ===
import os, sys
from openai import OpenAI

def get_api_key():
    k = API_KEY.strip()
    if not k or k.startswith("<在这里"):
        k = os.environ.get("ARK_API_KEY", "").strip()
    return k

def main():
    key = get_api_key()
    if not key:
        print("未找到 API Key：请在脚本顶部填写 API_KEY，或先 `export ARK_API_KEY=...`")
        sys.exit(1)

    client = OpenAI(base_url=ARK_BASE_URL, api_key=key)

    prompt = "你好！用一句话介绍你自己。"
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="medium",   # 可保留；若不支持会被忽略/报错
    )

    msg = resp.choices[0].message
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        print("----- reasoning -----")
        print(msg.reasoning_content)
    print("----- answer -----")
    print(msg.content)

    if getattr(resp, "usage", None):
        print("----- usage -----")
        print(resp.usage)

if __name__ == "__main__":
    main()
