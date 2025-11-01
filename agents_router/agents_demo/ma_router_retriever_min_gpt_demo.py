# 成功运行
#  -*- coding: utf-8 -*-
# Minimal Router + Retriever using OpenAI Agents SDK (ChatAnyWhere-compatible)

import asyncio, sys, json, difflib
from typing import List, Dict
from openai import AsyncOpenAI
from agents import (
    Agent, Runner, function_tool,
    set_default_openai_client, set_default_openai_api, set_tracing_disabled,
)

# ===== ① 你的网关与密钥（保持你能跑通的设置）=====
BASE_URL = "https://api.chatanywhere.tech/v1"
API_KEY  = "sk-AhGuNmK6xnFGdBCkFGpG0lcqj3TgLT7dQKU5JUSpaNQkUpZV"
MODEL    = "gpt-4o-mini"     # 用 /v1/models 里存在的模型
# ================================================

set_tracing_disabled(True)
set_default_openai_api("chat_completions")
set_default_openai_client(AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY))

# ===== ② 简单记录库（先内存，后面可换 JSON/DB/向量检索）=====
RECORDS: List[Dict] = [
    {"task": "open_laptop",        "ckpt": "act_ckpt_v1", "success_rate": 0.82, "notes": "抓取角度敏感"},
    {"task": "open_laptop",        "ckpt": "dp_ckpt_v3",  "success_rate": 0.76, "notes": "对光照更稳"},
    {"task": "beat_block_hammer",  "ckpt": "dp_ckpt_v5",  "success_rate": 0.71, "notes": "需更高频视觉"},
    {"task": "place_container_plate","ckpt":"act_ckpt_v2","success_rate": 0.65, "notes": "控制器需微调"},
]

def _match_score(q: str, rec: Dict) -> float:
    text = f"{rec['task']} {rec['ckpt']} {rec.get('notes','')}".lower()
    return difflib.SequenceMatcher(None, q.lower(), text).ratio()

@function_tool
def search_records(query: str, top_k: int = 3) -> str:
    """
    在本地记录库中检索与 query 最相关的 top_k 条记录，返回 JSON 字符串。
    """
    scored = sorted(RECORDS, key=lambda r: _match_score(query, r), reverse=True)[:top_k]
    return json.dumps(scored, ensure_ascii=False)

# ===== ③ Retriever：负责用工具检索，并给出“结构化结果 + 推荐建议”=====
retriever = Agent(
    name="Retriever",
    instructions=(
        "你是检索员：当用户询问某任务选哪个 ckpt/成功率等，"
        "1) 调用 search_records(query, top_k=3)；"
        "2) 解析返回的 JSON，按 success_rate 降序列出 Top-K；"
        "3) 选择 success_rate 最高者作为推荐，并用一句中文给出理由；"
        "4) 最后附上原始 JSON 引用。"
    ),
    tools=[search_records],
    model=MODEL,
)

# ===== ④ Router：根据问题是否涉及“选 ckpt/成功率/记录”，handoff 给 Retriever =====
router = Agent(
    name="Router",
    instructions=(
        "你是分诊路由：如果问题涉及“选择模型/ckpt/成功率/记录/检索/推荐策略”等关键词，"
        "请 handoff 给 Retriever；否则用一句中文简要回答。"
    ),
    handoffs=[retriever],
    model=MODEL,
)

# ===== ⑤ 运行示例 =====
async def main():
    # 支持命令行自定义问题；未传则默认问 open_laptop
    q = "对于 open_laptop 任务，应该选哪个 ckpt？给出理由与Top-K。" if len(sys.argv) == 1 else " ".join(sys.argv[1:])
    result = await Runner.run(router, input=q)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
