#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 Ops-MM-embedding-v1-2B 本地模型提取多模态embedding
从20个任务的视频中提取前50个episode的首帧并生成embedding
使用GPU 7
输出到新文件夹：mm_embed_cache_50videos
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
VIDEO_PER_TASK = 50  # 每个任务处理50个视频（全部）
EMBEDDING_DIM = 1536  # Ops-MM-embedding-v1-2B 的维度

# 20个任务配置：任务名 -> 指令
TASK_CONFIGS = {
    # 原有的5个任务
    "adjust_bottle": "Adjust the bottle alignment.",
    "beat_block_hammer": "Beat the block after grabbing the hammer.",
    "click_alarmclock": "Click the alarm clock button.",
    "open_laptop": "Open the laptop.",
    "place_container_plate": "Place the container onto the plate.",
    
    # 新增的15个任务
    "click_bell": "Click the bell.",
    "dump_bin_bigbin": "Dump items into the big bin.",
    "grab_roller": "Grab the roller.",
    "handover_mic": "Handover the microphone.",
    "lift_pot": "Lift the pot.",
    "move_can_pot": "Move the can to the pot.",
    "move_playingcard_away": "Move the playing card away.",
    "pick_dual_bottles": "Pick up dual bottles.",
    "place_burger_fries": "Place burger and fries.",
    "place_empty_cup": "Place the empty cup.",
    "put_bottles_dustbin": "Put bottles into the dustbin.",
    "put_object_cabinet": "Put object into the cabinet.",
    "shake_bottle": "Shake the bottle.",
    "stack_bowls_three": "Stack three bowls.",
    "stack_bowls_two": "Stack two bowls.",
}

# 视频数据基础目录
VIDEO_BASE_DIR = "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/video_test/from_dataset"

# 输出目录（新文件夹）
CACHE_BASE_DIR = "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache_50videos"
FRAMES_DIR = f"{CACHE_BASE_DIR}/frames"
EMBEDS_DIR = f"{CACHE_BASE_DIR}/embeddings"
FIGURES_DIR = f"{CACHE_BASE_DIR}/figures"

# ===================================

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _l2_norm(vec):
    return math.sqrt(sum((float(x) ** 2 for x in vec)))

def extract_first_frame(video_path: str, out_img_path: str):
    """提取视频首帧为图片文件"""
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

    try:
        import imageio.v2 as iio
        reader = iio.get_reader(video_path)
        frame0 = reader.get_data(0)
        iio.imwrite(out_img_path, frame0)
        return out_img_path
    except Exception:
        pass

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

def _list_task_videos(task_dir: str, limit: int):
    """在任务目录下匹配 episode*.mp4，按数字排序取前 limit 个"""
    patt = os.path.join(task_dir, "episode*.mp4")
    files = sorted(glob.glob(patt), key=lambda x: int(os.path.basename(x).replace('episode', '').replace('.mp4', '')))
    return files[:max(0, int(limit))]

class LocalMMEmbedding:
    """使用本地 Ops-MM-embedding-v1-2B 模型"""
    
    def __init__(self, model_path: str):
        print(f"[Info] 正在加载本地模型: {model_path}")
        print(f"[Info] 使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
        
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
        """生成文本+图片的融合embedding"""
        try:
            from PIL import Image
            
            image = Image.open(image_path).convert('RGB')
            instruction = "Generate a unified embedding for the robot manipulation task."
            embeddings = self.model.get_fused_embeddings(
                texts=[text],
                images=[[image]],
                instruction=instruction
            )
            
            if isinstance(embeddings, torch.Tensor):
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
    _ensure_dir(FIGURES_DIR)
    
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
        "num_tasks": len(TASK_CONFIGS),
        "gpu": os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A'),
        "items": []
    }
    
    total = 0
    print(f"\n[Info] 批次开始: {batch_ts}")
    print(f"[Info] 每个任务处理前 {VIDEO_PER_TASK} 个视频")
    print(f"[Info] 共 {len(TASK_CONFIGS)} 个任务")
    print(f"[Info] 预计生成 {len(TASK_CONFIGS) * VIDEO_PER_TASK} 个embedding")
    print("="*70 + "\n")
    
    for task_idx, (task_key, instruction) in enumerate(TASK_CONFIGS.items(), 1):
        print(f"\n{'='*70}")
        print(f"任务 {task_idx}/{len(TASK_CONFIGS)}: {task_key}")
        print(f"指令: {instruction}")
        print(f"{'='*70}")
        
        task_dir = os.path.join(VIDEO_BASE_DIR, task_key)
        
        if not os.path.isdir(task_dir):
            print(f"[Skip] 任务目录不存在: {task_dir}")
            continue
        
        video_list = _list_task_videos(task_dir, VIDEO_PER_TASK)
        if not video_list:
            print(f"[Skip] 未找到任何 episode: {task_dir}")
            continue
        
        print(f"[Info] 找到 {len(video_list)} 个视频文件")
        
        for vid_idx, vpath in enumerate(video_list, 1):
            if not os.path.isfile(vpath):
                print(f"[Skip] 视频不存在: {vpath}")
                continue
            
            episode_stem = os.path.splitext(os.path.basename(vpath))[0]
            
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
        
        print(f"[Info] 任务 {task_key} 完成: {len([x for x in manifest['items'] if x['task'] == task_key])} 个样本")
    
    # 5) 保存清单
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ 批次完成！")
    print(f"{'='*70}")
    print(f"清单文件: {manifest_path}")
    print(f"共生成 {total} 条记录（目标: {VIDEO_PER_TASK} × {len(TASK_CONFIGS)} = {VIDEO_PER_TASK * len(TASK_CONFIGS)}）")
    print(f"Embeddings 保存在: {EMBEDS_DIR}")
    print(f"首帧图片保存在: {FRAMES_DIR}")
    print(f"图像将保存在: {FIGURES_DIR}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

