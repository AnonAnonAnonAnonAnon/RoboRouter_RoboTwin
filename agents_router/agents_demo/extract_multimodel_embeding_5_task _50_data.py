#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#未实现

"""
embed_batch_from_videos.py
从多个“视频+指令”对中：
1) 提取视频首帧为图片
2) 生成（指令+首帧）的多模态嵌入
3) 落盘：首帧图片、逐条 embedding JSON、以及一个批次 manifest JSON
"""

# ========= 配置区（按需修改） =========
# -- 嵌入配置（保持与你项目一致） --
VOYAGE_API_KEY     = "pa-tEigYTRrvWMOSB2WIcH4f6FkhzEwF8xZ3kUHbFG4hz9"  # 按你的要求，常量明文
VOYAGE_MODEL       = "voyage-multimodal-3"
EMBEDDING_DIM      = 1024
EMBEDDING_BACKEND  = "voyage"  # 可改为 "clip" 做本地快速调试（需 transformers/torch）

# -- 数据源：任务 -> (指令, 视频路径) --
VIDEOS = {
    "adjust_bottle": (
        "Adjust the bottle alignment.",
        "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/adjust_bottle/episode0.mp4",
    ),
    "beat_block_hammer": (
        "Beat the block after grabbing the hammer.",
        "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/beat_block_hammer/episode0.mp4",
    ),
    "click_alarmclock": (
        "Click the alarm clock button.",
        "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/click_alarmclock/episode0.mp4",
    ),
    "open_laptop": (
        "Open the laptop.",
        "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/open_laptop/episode0.mp4",
    ),
    "place_container_plate": (
        "Place the container onto the plate.",
        "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/place_container_plate/episode0.mp4",
    ),
}

# -- 输出目录（会自动创建） --
CACHE_BASE_DIR = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache"
FRAMES_DIR     = f"{CACHE_BASE_DIR}/frames"
EMBEDS_DIR     = f"{CACHE_BASE_DIR}/embeddings"
# ===================================

import os, json, math, subprocess
from datetime import datetime

# 复用你项目里的多模态嵌入封装
from mutil_rag_demo.rag_retriever import MultimodalEmbedding

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _l2_norm(vec):
    return math.sqrt(sum((float(x) ** 2 for x in vec)))

def extract_first_frame(video_path: str, out_img_path: str):
    """
    提取视频首帧为图片文件。
    优先用 OpenCV；若不可用则尝试 imageio；再不行用 ffmpeg 命令。
    """
    # 尝试 OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("cv2.VideoCapture open failed")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("cv2 read first frame failed")
        cv2.imwrite(out_img_path, frame)
        return out_img_path
    except Exception:
        pass

    # 尝试 imageio
    try:
        import imageio.v2 as iio
        reader = iio.get_reader(video_path)
        frame0 = reader.get_data(0)
        iio.imwrite(out_img_path, frame0)
        return out_img_path
    except Exception:
        pass

    # 尝试 ffmpeg
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", out_img_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        if os.path.isfile(out_img_path):
            return out_img_path
        raise RuntimeError("ffmpeg did not produce output")
    except Exception as e:
        raise RuntimeError(f"extract_first_frame failed for {video_path}: {e}")

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
        raise RuntimeError("未找到 VOYAGE_API_KEY：请在配置区提供。")

    # 准备目录和批次时间戳
    _ensure_dir(CACHE_BASE_DIR); _ensure_dir(FRAMES_DIR); _ensure_dir(EMBEDS_DIR)
    batch_ts = datetime.now().strftime("%m%d%H%M%S")
    manifest_path = os.path.join(CACHE_BASE_DIR, f"manifest_{batch_ts}.json")

    manifest = {
        "batch_timestamp": batch_ts,
        "model": VOYAGE_MODEL,
        "backend": EMBEDDING_BACKEND,
        "embedding_dim": EMBEDDING_DIM,
        "items": []
    }

    print(f"[Info] 批次开始: {batch_ts}")
    for task_key, (instruction, vpath) in VIDEOS.items():
        if not os.path.isfile(vpath):
            print(f"[Skip] 视频不存在: {vpath}")
            continue

        # 1) 提取首帧
        frame_name = f"{task_key}_{batch_ts}.jpg"
        frame_path = os.path.join(FRAMES_DIR, frame_name)
        print(f"[Info] 提取首帧: {task_key} -> {frame_path}")
        try:
            extract_first_frame(vpath, frame_path)
        except Exception as e:
            print(f"[Warn] 提取首帧失败: {e}")
            continue

        # 2) 生成多模态嵌入
        print(f"[Info] 生成嵌入: {task_key}")
        vec = get_mm_embedding(instruction, frame_path)
        if not isinstance(vec, (list, tuple)) or len(vec) == 0:
            print(f"[Warn] 嵌入结果为空，跳过: {task_key}")
            continue

        # 3) 保存逐条 embedding
        emb_name = f"mm_embed_{task_key}_{batch_ts}.json"
        emb_path = os.path.join(EMBEDS_DIR, emb_name)
        rec = {
            "task": task_key,
            "instruction": instruction,
            "video_path": vpath,
            "frame_path": frame_path,
            "model": VOYAGE_MODEL,
            "backend": EMBEDDING_BACKEND,
            "embedding_dim": len(vec),
            "vector": [float(x) for x in vec],
            "l2_norm": _l2_norm(vec),
            "timestamp": batch_ts,
        }
        with open(emb_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {emb_path}")

        # 4) 记入 manifest
        manifest["items"].append({
            "task": task_key,
            "instruction": instruction,
            "video_path": vpath,
            "frame_path": frame_path,
            "embedding_path": emb_path,
            "l2_norm": rec["l2_norm"],
        })

    # 5) 保存清单
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] 批次完成，清单: {manifest_path}")
    print(f"[OK] 共生成 {len(manifest['items'])} 条记录")

if __name__ == "__main__":
    main()
