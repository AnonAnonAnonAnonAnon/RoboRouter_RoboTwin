# doubao 测试,单张图片问答
#成功

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === 配置区（按需修改） ===
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_ID = "doubao-seed-1-6-251015"      # 注意不要带换行/空格
API_KEY = "70f0d563-91a5-4704-a00e-f00cf3a9c864"             # 或留空，用环境变量 ARK_API_KEY
IMAGE_PATH = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents/frames_to_push/f_0.jpg"
QUESTION = "请说明这张图片的主要内容，并列出3个关键要素。"

# === 代码区（一般无需改动） ===
import os, sys, base64, mimetypes
from openai import OpenAI

def get_api_key():
    k = (API_KEY or "").strip()
    if not k or k.startswith("<你的"):
        k = os.environ.get("ARK_API_KEY", "").strip()
    return k

def to_data_url(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到图片文件：{path}")
    mime, _ = mimetypes.guess_type(path)
    # 常见图片类型兜底
    if mime is None:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif ext in [".png"]:
            mime = "image/png"
        elif ext in [".webp"]:
            mime = "image/webp"
        else:
            raise ValueError(f"无法识别的图片类型：{ext}（建议使用 jpg/png/webp）")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def main():
    key = get_api_key()
    if not key:
        print("未找到 API Key：请在脚本顶部填写 API_KEY，或先 `export ARK_API_KEY=...`")
        sys.exit(1)

    img_data_url = to_data_url(IMAGE_PATH)

    client = OpenAI(base_url=ARK_BASE_URL, api_key=key)
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_data_url}},
                {"type": "text", "text": QUESTION},
            ],
        }],
        reasoning_effort="medium",
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
