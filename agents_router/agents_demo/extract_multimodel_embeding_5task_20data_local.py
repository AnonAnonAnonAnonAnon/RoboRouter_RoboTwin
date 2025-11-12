#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 Ops-MM-embedding-v1-2B 本地模型提取多模态embedding
从5个任务的视频中提取前20个episode的首帧并生成embedding
使用GPU 7
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'  # 使用第7张GPU卡

import sys
import json
import math
import subprocess
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import torch

# ========= 配置区 =========
MODEL_PATH = "/data/work/public/llm_modles/Ops-MM-embedding-v1-2B"

# 将模型目录添加到Python路径以便导入ops_mm_embedding_v1
sys.path.insert(0, MODEL_PATH)

# 导入本地模型
from ops_mm_embedding_v1 import OpsMMEmbeddingV1
VIDEO_PER_TASK = 20  # 每个任务处理20个视频
EMBEDDING_DIM = 1024  # Ops-MM-embedding-v1-2B 的维度

# 数据源：任务 -> (指令, 某个该任务视频的路径)
VIDEOS = {
    "adjust_bottle": (
        "Adjust the bottle alignment.",
        "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/adjust_bottle/episode0.mp4",
    ),
    "beat_block_hammer": (
        "Beat the block after grabbing the hammer.",
        "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/beat_block_hammer/episode0.mp4",
    ),
    "click_alarmclock": (
        "Click the alarm clock button.",
        "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/click_alarmclock/episode0.mp4",
    ),
    "open_laptop": (
        "Open the laptop.",
        "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/open_laptop/episode0.mp4",
    ),
    "place_container_plate": (
        "Place the container onto the plate.",
        "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/place_container_plate/episode0.mp4",
    ),
}

# 输出目录
CACHE_BASE_DIR = "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache"
FRAMES_DIR = f"{CACHE_BASE_DIR}/frames"
EMBEDS_DIR = f"{CACHE_BASE_DIR}/embeddings"

# ===================================

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

def _list_task_videos(seed_path: str, limit: int):
    """
    给定某个任务的任意一个 episode 路径，在同目录下匹配 episode*.mp4，
    按数字排序取前 limit 个。
    """
    d = os.path.dirname(seed_path)
    patt = os.path.join(d, "episode*.mp4")
    files = sorted(glob.glob(patt), key=lambda x: int(os.path.basename(x).replace('episode', '').replace('.mp4', '')))
    if not files and os.path.isfile(seed_path):
        files = [seed_path]
    return files[:max(0, int(limit))]

class LocalMMEmbedding:
    """使用本地 Ops-MM-embedding-v1-2B 模型"""
    
    def __init__(self, model_path: str):
        print(f"[Info] 正在加载本地模型: {model_path}")
        print(f"[Info] 使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
        
        # 尝试使用flash_attention_2，失败则使用eager
        try:
            print(f"[Info] 尝试使用 flash_attention_2...")
            self.model = OpsMMEmbeddingV1(
                model_path,
                device="cuda",
                attn_implementation="flash_attention_2"
            )
            print(f"[Info] ✓ 使用 flash_attention_2")
        except ImportError as e:
            print(f"[Warning] flash_attention_2 不可用，使用 eager attention")
            self.model = OpsMMEmbeddingV1(
                model_path,
                device="cuda",
                attn_implementation="eager"
            )
        print(f"[Info] ✓ 模型加载完成")
    
    def get_embedding(self, text: str, image_path: str) -> List[float]:
        """
        生成文本+图片的融合embedding
        """
        try:
            from PIL import Image
            
            # 读取图片
            image = Image.open(image_path).convert('RGB')
            
            # 使用 get_fused_embeddings 获取文本+图片的融合表征
            instruction = "Generate a unified embedding for the robot manipulation task."
            embeddings = self.model.get_fused_embeddings(
                texts=[text],
                images=[[image]],
                instruction=instruction
            )
            
            # 转换为列表（处理BFloat16类型）
            if isinstance(embeddings, torch.Tensor):
                # 如果是BFloat16，先转换为float32
                emb_tensor = embeddings[0]
                if emb_tensor.dtype == torch.bfloat16:
                    emb_tensor = emb_tensor.float()
                embedding_vec = emb_tensor.cpu().numpy().tolist()
            else:
                embedding_vec = embeddings[0].tolist()
            
            return embedding_vec
            
        except Exception as e:
            print(f"[Error] 生成embedding失败: {e}")
            import traceback
            traceback.print_exc()
            return []

def main():
    # 准备目录和批次时间戳
    _ensure_dir(CACHE_BASE_DIR)
    _ensure_dir(FRAMES_DIR)
    _ensure_dir(EMBEDS_DIR)
    
    batch_ts = datetime.now().strftime("%m%d%H%M%S")
    manifest_path = os.path.join(CACHE_BASE_DIR, f"manifest_{batch_ts}.json")
    
    # 初始化本地模型
    print("\n" + "="*70)
    print("初始化 Ops-MM-embedding-v1-2B 本地模型")
    print("="*70)
    embedder = LocalMMEmbedding(MODEL_PATH)
    
    manifest = {
        "batch_timestamp": batch_ts,
        "model": "Ops-MM-embedding-v1-2B",
        "model_path": MODEL_PATH,
        "backend": "local",
        "embedding_dim": EMBEDDING_DIM,
        "video_per_task": VIDEO_PER_TASK,
        "gpu": os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A'),
        "items": []
    }
    
    total = 0
    print(f"\n[Info] 批次开始: {batch_ts}")
    print(f"[Info] 每个任务处理前 {VIDEO_PER_TASK} 个视频")
    print(f"[Info] 共 {len(VIDEOS)} 个任务")
    print("="*70 + "\n")
    
    for task_idx, (task_key, (instruction, seed_vpath)) in enumerate(VIDEOS.items(), 1):
        print(f"\n{'='*70}")
        print(f"任务 {task_idx}/{len(VIDEOS)}: {task_key}")
        print(f"指令: {instruction}")
        print(f"{'='*70}")
        
        # 列出该任务的前 N 个 episode
        video_list = _list_task_videos(seed_vpath, VIDEO_PER_TASK)
        if not video_list:
            print(f"[Skip] 未找到任何 episode: {seed_vpath}")
            continue
        
        print(f"[Info] 找到 {len(video_list)} 个视频文件")
        
        for vid_idx, vpath in enumerate(video_list, 1):
            if not os.path.isfile(vpath):
                print(f"[Skip] 视频不存在: {vpath}")
                continue
            
            episode_stem = os.path.splitext(os.path.basename(vpath))[0]  # e.g., episode0
            
            # 1) 提取首帧
            frame_name = f"{task_key}_{episode_stem}_{batch_ts}.jpg"
            frame_path = os.path.join(FRAMES_DIR, frame_name)
            
            print(f"  [{vid_idx}/{len(video_list)}] 提取首帧: {episode_stem}", end=" ... ")
            try:
                extract_first_frame(vpath, frame_path)
                print("✓")
            except Exception as e:
                print(f"✗ 失败: {e}")
                continue
            
            # 2) 生成多模态嵌入
            print(f"  [{vid_idx}/{len(video_list)}] 生成嵌入: {episode_stem}", end=" ... ")
            vec = embedder.get_embedding(instruction, frame_path)
            if not isinstance(vec, (list, tuple)) or len(vec) == 0:
                print(f"✗ 嵌入结果为空")
                continue
            print(f"✓ (dim={len(vec)})")
            
            # 3) 保存逐条 embedding
            emb_name = f"mm_embed_{task_key}_{episode_stem}_{batch_ts}.json"
            emb_path = os.path.join(EMBEDS_DIR, emb_name)
            rec = {
                "task": task_key,
                "episode": episode_stem,
                "instruction": instruction,
                "video_path": vpath,
                "frame_path": frame_path,
                "model": "Ops-MM-embedding-v1-2B",
                "backend": "local",
                "embedding_dim": len(vec),
                "vector": [float(x) for x in vec],
                "l2_norm": _l2_norm(vec),
                "timestamp": batch_ts,
            }
            with open(emb_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            
            # 4) 记入 manifest
            manifest["items"].append({
                "task": task_key,
                "episode": episode_stem,
                "instruction": instruction,
                "video_path": vpath,
                "frame_path": frame_path,
                "embedding_path": emb_path,
                "l2_norm": rec["l2_norm"],
            })
            total += 1
    
    # 5) 保存清单
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ 批次完成！")
    print(f"{'='*70}")
    print(f"清单文件: {manifest_path}")
    print(f"共生成 {total} 条记录（目标: {VIDEO_PER_TASK} × {len(VIDEOS)} = {VIDEO_PER_TASK * len(VIDEOS)}）")
    print(f"Embeddings 保存在: {EMBEDS_DIR}")
    print(f"首帧图片保存在: {FRAMES_DIR}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

