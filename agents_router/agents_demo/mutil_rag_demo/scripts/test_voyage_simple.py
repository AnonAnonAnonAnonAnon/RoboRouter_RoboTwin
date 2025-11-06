#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试Voyage AI多模态Embedding（图片+文字）
"""

import os
import sys
from pathlib import Path

# 添加父目录到Python路径
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from rag_retriever import MultimodalEmbedding

# API Key
API_KEY = "pa-tEigYTRrvWMOSB2WIcH4f6FkhzEwF8xZ3kUHbFG4hz9"

# 测试图片路径
IMAGE_PATH = "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/frames_to_push/f_0.jpg"

# 测试文本
TEXT = "机器人正在执行抓取任务"

print("="*60)
print("测试 Voyage AI 多模态Embedding (图片+文字)")
print("="*60)

# 检查图片是否存在
print(f"\n检查图片...")
if not os.path.exists(IMAGE_PATH):
    print(f"✗ 错误：图片不存在！")
    print(f"   路径: {IMAGE_PATH}")
    print(f"\n请确保图片存在，或修改IMAGE_PATH变量")
    sys.exit(1)

from PIL import Image
img = Image.open(IMAGE_PATH)
print(f"✓ 图片存在")
print(f"   路径: {IMAGE_PATH}")
print(f"   尺寸: {img.size}")
print(f"   格式: {img.format}")

# 初始化
print("\n1. 初始化embedding生成器...")
emb = MultimodalEmbedding(
    api_key=API_KEY,
    model="voyage-multimodal-3",
    embedding_dim=1024,
    backend="voyage"
)
print("✓ 初始化成功")

# 测试多模态embedding（图片+文字）
print("\n2. 测试多模态embedding (图片+文字)...")
print(f"   文本: {TEXT}")
print(f"   图片: {IMAGE_PATH}")
print("   正在调用Voyage AI API...")

try:
    result = emb.get_embedding(TEXT, IMAGE_PATH)
    print(f"✓ 成功！得到 {len(result)} 维向量")
    print(f"   前10个值: {result[:10]}")
    
    # 计算向量的范数（验证向量有效性）
    import math
    norm = math.sqrt(sum(x**2 for x in result))
    print(f"   向量范数: {norm:.4f}")
    
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("🎉 测试通过！Voyage AI 多模态embedding工作正常")
print("="*60)
print("\n下一步:")
print("  1. 启动Qdrant: cd mutil_rag_demo && ./00start_qdrant.sh")
print("  2. 插入数据: python 01_setup_database.py")
print("  3. 执行检索: python 02_simple_search.py -q '打开笔记本' -i '图片路径'")

