#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤1: 数据库初始化和数据插入脚本
功能：将文本记录embedding后插入Qdrant向量数据库
"""

import os
import sys
from typing import List, Dict

# 导入本地模块
from vector_db import VectorDB
from rag_retriever import MultimodalEmbedding, RAGRetriever


# ===== 配置 =====
# Qdrant配置
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "robot_task_records"

# Embedding配置
# 支持的backend: "voyage", "jina", "clip"
EMBEDDING_BACKEND = "voyage"  # 默认使用Voyage AI
EMBEDDING_API_KEY = "pa-tEigYTRrvWMOSB2WIcH4f6FkhzEwF8xZ3kUHbFG4hz9"  # Voyage AI API Key
EMBEDDING_MODEL = "voyage-multimodal-3"  # Voyage: voyage-multimodal-3, Jina: jina-clip-v1
EMBEDDING_DIM = 1024  # voyage-multimodal-3: 1024, jina-clip-v1: 768


# ===== 数据定义 =====
# 这里定义你要存入数据库的记录
RECORDS: List[Dict] = [
    {
        "id": 1,
        "task": "open_laptop",
        "ckpt": "act_ckpt_v1",
        "success_rate": 0.82,
        "notes": "抓取角度敏感，需要精确定位铰链位置",
        "description": "使用ACT策略打开笔记本电脑，成功率82%。关键点：抓取角度敏感，需要精确定位铰链位置。适合光照稳定的环境。"
    },
    {
        "id": 2,
        "task": "open_laptop",
        "ckpt": "dp_ckpt_v3",
        "success_rate": 0.76,
        "notes": "对光照更稳定，鲁棒性好",
        "description": "使用扩散策略打开笔记本电脑，成功率76%。特点：对光照变化更稳定，鲁棒性好，适合复杂环境。"
    },
    {
        "id": 3,
        "task": "beat_block_hammer",
        "ckpt": "dp_ckpt_v5",
        "success_rate": 0.71,
        "notes": "需更高频视觉反馈",
        "description": "使用扩散策略v5抓取锤子敲击木块，成功率71%。需要更高频率的视觉反馈来调整轨迹，对力控要求较高。"
    },
    {
        "id": 4,
        "task": "beat_block_hammer",
        "ckpt": "act_ckpt_v4",
        "success_rate": 0.68,
        "notes": "轨迹平滑但精度略低",
        "description": "使用ACT策略v4完成锤子敲击任务，成功率68%。轨迹平滑，但终点精度略低，适合对位置要求不严格的场景。"
    },
    {
        "id": 5,
        "task": "place_container_plate",
        "ckpt": "act_ckpt_v2",
        "success_rate": 0.65,
        "notes": "控制器需微调，对容器形状敏感",
        "description": "将容器放置到盘子上，使用ACT策略v2，成功率65%。控制器需要针对不同容器形状进行微调。"
    },
    {
        "id": 6,
        "task": "grasp_small_object",
        "ckpt": "dp_ckpt_v6",
        "success_rate": 0.88,
        "notes": "精细抓取，适合小物体",
        "description": "抓取小型物体（如螺丝、硬币），使用扩散策略v6，成功率88%。专门优化了精细抓取能力，夹持力控制精准。"
    },
    {
        "id": 7,
        "task": "pick_and_place",
        "ckpt": "act_ckpt_v3",
        "success_rate": 0.79,
        "notes": "通用抓放任务",
        "description": "通用的抓取和放置任务，使用ACT策略v3，成功率79%。适合大多数物体，泛化性能好。"
    },
    {
        "id": 8,
        "task": "push_object",
        "ckpt": "dp_ckpt_v4",
        "success_rate": 0.73,
        "notes": "推动物体到目标位置",
        "description": "推动物体到指定位置，使用扩散策略v4，成功率73%。对摩擦力和力度控制要求较高。"
    },
]


def check_qdrant_connection():
    """检查Qdrant服务是否可用"""
    import requests
    try:
        response = requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}/", timeout=3)
        if response.status_code == 200:
            print(f"✓ Qdrant服务运行正常: {QDRANT_HOST}:{QDRANT_PORT}")
            return True
    except Exception as e:
        print(f"✗ 无法连接到Qdrant服务: {e}")
        print(f"\n请先启动Qdrant:")
        print(f"  ./start_qdrant.sh")
        print(f"  或运行: docker run -d -p 6333:6333 qdrant/qdrant")
        return False


def setup_database():
    """初始化数据库并插入数据"""
    print("\n" + "="*60)
    print("  向量数据库初始化和数据插入")
    print("="*60)
    print()
    
    # 检查Qdrant连接
    if not check_qdrant_connection():
        sys.exit(1)
    
    # 检查API Key
    if EMBEDDING_BACKEND in ["voyage", "jina"]:
        if EMBEDDING_API_KEY in ["voyage_xxx", "jina_xxx"]:
            print(f"\n⚠ 警告: 未设置 {EMBEDDING_BACKEND.upper()}_API_KEY")
            print("将使用随机向量模拟（仅用于测试）")
            print("要使用真实embedding，请设置:")
            if EMBEDDING_BACKEND == "voyage":
                print("  export VOYAGE_API_KEY='your_key'")
                print("  获取API Key: https://www.voyageai.com/")
            else:
                print("  export JINA_API_KEY='your_key'")
                print("  获取API Key: https://jina.ai/")
            print()
            response = input("继续吗？(y/n): ")
            if response.lower() != 'y':
                print("已取消")
                sys.exit(0)
    
    # 1. 创建向量数据库连接
    print("[1/4] 连接向量数据库...")
    try:
        vector_db = VectorDB(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=COLLECTION_NAME,
            embedding_dim=EMBEDDING_DIM
        )
        print(f"  ✓ 已连接: {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"  ✓ Collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        sys.exit(1)
    
    # 2. 创建Embedding生成器
    print("\n[2/4] 初始化Embedding生成器...")
    embedding_gen = MultimodalEmbedding(
        api_key=EMBEDDING_API_KEY,
        model=EMBEDDING_MODEL,
        embedding_dim=EMBEDDING_DIM,
        backend=EMBEDDING_BACKEND
    )
    backend_name = {
        "voyage": "Voyage AI (Multimodal-3)",
        "jina": "Jina AI (CLIP)",
        "clip": "本地CLIP"
    }.get(EMBEDDING_BACKEND, EMBEDDING_BACKEND)
    print(f"  ✓ 使用backend: {backend_name}")
    print(f"  ✓ 模型: {EMBEDDING_MODEL}")
    print(f"  ✓ 向量维度: {EMBEDDING_DIM}")
    
    # 3. 创建RAG检索器
    print("\n[3/4] 创建RAG检索器...")
    rag_retriever = RAGRetriever(vector_db, embedding_gen)
    print("  ✓ RAG检索器创建成功")
    
    # 4. 插入数据
    print(f"\n[4/4] 插入{len(RECORDS)}条记录...")
    print("  (正在生成embedding，可能需要几秒钟...)")
    
    try:
        rag_retriever.insert_text_records(RECORDS)
        print(f"  ✓ 成功插入{len(RECORDS)}条记录")
    except Exception as e:
        print(f"  ✗ 插入失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 5. 验证
    print("\n[验证] 检查数据库状态...")
    count = vector_db.get_count()
    print(f"  ✓ 数据库中共有 {count} 条记录")
    
    # 6. 快速测试
    print("\n[测试] 执行快速检索测试...")
    test_query = "打开笔记本电脑"
    results = rag_retriever.search(test_query, top_k=2)
    print(f"  查询: '{test_query}'")
    print(f"  结果:")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r['task']} ({r['ckpt']}) - 相似度:{r['score']:.4f}")
    
    print("\n" + "="*60)
    print("✓✓✓ 数据库初始化完成！✓✓✓")
    print("="*60)
    print("\n下一步:")
    print("  运行检索脚本: python 02_simple_search.py")
    print("  或运行完整Agent: python ma_router_retriever_multimodal_qdrant.py")
    print()


def add_more_records(new_records: List[Dict]):
    """追加更多记录到数据库"""
    print(f"\n追加 {len(new_records)} 条新记录...")
    
    vector_db = VectorDB(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
        embedding_dim=EMBEDDING_DIM
    )
    
    embedding_gen = MultimodalEmbedding(
        api_key=EMBEDDING_API_KEY,
        model=EMBEDDING_MODEL,
        embedding_dim=EMBEDDING_DIM,
        backend=EMBEDDING_BACKEND
    )
    
    rag_retriever = RAGRetriever(vector_db, embedding_gen)
    rag_retriever.insert_text_records(new_records)
    
    print(f"✓ 追加完成，当前总数: {vector_db.get_count()}")


def clear_database():
    """清空数据库"""
    print("\n⚠ 警告: 即将删除collection")
    response = input(f"确认删除 '{COLLECTION_NAME}'？(yes/no): ")
    if response.lower() == 'yes':
        vector_db = VectorDB(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=COLLECTION_NAME,
            embedding_dim=EMBEDDING_DIM
        )
        vector_db.delete_collection()
        print("✓ Collection已删除")
    else:
        print("已取消")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="向量数据库初始化和数据管理")
    parser.add_argument(
        "action",
        nargs="?",
        default="setup",
        choices=["setup", "clear", "check"],
        help="操作: setup(初始化), clear(清空), check(检查状态)"
    )
    
    args = parser.parse_args()
    
    if args.action == "setup":
        setup_database()
    elif args.action == "clear":
        clear_database()
    elif args.action == "check":
        if check_qdrant_connection():
            vector_db = VectorDB(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                collection_name=COLLECTION_NAME,
                embedding_dim=EMBEDDING_DIM
            )
            count = vector_db.get_count()
            print(f"✓ Collection '{COLLECTION_NAME}' 有 {count} 条记录")

