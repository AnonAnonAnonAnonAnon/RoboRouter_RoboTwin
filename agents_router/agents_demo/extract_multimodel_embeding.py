#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embed_one_pair_mm.py
最小实现：对“指令+图片”生成 Voyage 多模态嵌入，并保存 JSON（含向量）到 cache。
"""

# ========= 配置区（按需修改） =========
VOYAGE_API_KEY = "pa-tEigYTRrvWMOSB2WIcH4f6FkhzEwF8xZ3kUHbFG4hz9"
VOYAGE_MODEL     = "voyage-multimodal-3"
EMBEDDING_DIM    = 1024
EMBEDDING_BACKEND = "voyage"

INSTRUCTION_TEXT = "Beat the block after grabbing the hammer."
IMAGE_PATH       = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/frames_to_push/f_0.jpg"

# 输出目录（保存 embedding JSON）
CACHE_DIR        = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache"

# ========= 代码区（一般无需改动） =========
import os, json, math
from datetime import datetime

# 复用你项目里的多模态嵌入封装
from mutil_rag_demo.rag_retriever import MultimodalEmbedding

def _l2_norm(vec):
    return math.sqrt(sum((float(x) ** 2 for x in vec)))

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def get_mm_embedding(text: str, image_path: str):
    """使用项目内的 MultimodalEmbedding 生成“文本+图片”的联合向量。"""
    embedder = MultimodalEmbedding(
        api_key=VOYAGE_API_KEY,
        model=VOYAGE_MODEL,
        embedding_dim=EMBEDDING_DIM,
        backend=EMBEDDING_BACKEND,
    )
    return embedder.get_embedding(text, image_path)

def main():
    if not VOYAGE_API_KEY:
        raise RuntimeError("未找到 VOYAGE_API_KEY：请在环境变量或配置区提供。")

    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"图片不存在：{IMAGE_PATH}")

    print("[Info] 生成多模态嵌入...")
    vec = get_mm_embedding(INSTRUCTION_TEXT, IMAGE_PATH)

    if not isinstance(vec, (list, tuple)) or len(vec) == 0:
        raise RuntimeError(f"嵌入结果异常：{type(vec)}")

    print(f"[OK] 向量维度：{len(vec)}（预期 {EMBEDDING_DIM}）")
    print(f"[OK] 前8维示例：{[round(float(x), 6) for x in vec[:8]]}")
    print(f"[OK] L2范数：{_l2_norm(vec):.4f}")

    _ensure_dir(CACHE_DIR)
    ts = datetime.now().strftime("%m%d%H%M%S")
    out_path = os.path.join(CACHE_DIR, f"mm_embed_{ts}.json")

    payload = {
        "model": VOYAGE_MODEL,
        "embedding_dim": len(vec),
        "instruction": INSTRUCTION_TEXT,
        "image_path": IMAGE_PATH,
        "vector": [float(x) for x in vec],        # 明确转 float，避免类型不兼容
        "l2_norm": _l2_norm(vec),
        "timestamp": ts,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[Saved]", out_path)

if __name__ == "__main__":
    main()
