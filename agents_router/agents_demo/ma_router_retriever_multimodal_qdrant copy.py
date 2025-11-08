# -*- coding: utf-8 -*-
"""
多模态向量检索系统 - Router + Retriever with Qdrant
- 使用 Qdrant 向量数据库
- 支持多模态 Embedding（图片+文本混合输入）
- 语义检索替代字符串匹配
- 数据库存储：文本记录的向量
- 查询：图片+文字 → 多模态embedding → 向量检索


查询示例：
  python ma_router_retriever_multimodal_qdrant.py \
    --query "根据当前场景，推荐适合的checkpoint完成抓取任务" \
    --image "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/frames_to_push/f_0.jpg"

  
注意：
- 查询时必须同时提供文本(--query)和图片(--image)
- 首次运行会自动从 robot_records_data.py 插入数据（约2-3分钟）
- 后续运行自动跳过插入
"""

import asyncio, sys, json, os, argparse
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from agents import (
    Agent, Runner, function_tool,
    set_default_openai_client, set_default_openai_api, set_tracing_disabled,
)

# 导入自定义模块（从mutil_rag_demo子目录）
from mutil_rag_demo.vector_db import VectorDB
from mutil_rag_demo.rag_retriever import MultimodalEmbedding, RAGRetriever

# 导入机器人任务记录数据
# from robot_records_data import RECORDS
from robot_records_data_read_from_txt import RECORDS


# ===== ① 配置区 =====
# OpenAI API配置
BASE_URL = "https://api.chatanywhere.tech/v1"
API_KEY = "sk-AhGuNmK6xnFGdBCkFGpG0lcqj3TgLT7dQKU5JUSpaNQkUpZV"
MODEL = "gpt-4o-mini"

# 多模态Embedding API配置（使用Voyage AI的多模态embedding）
VOYAGE_API_KEY = "pa-tEigYTRrvWMOSB2WIcH4f6FkhzEwF8xZ3kUHbFG4hz9"
VOYAGE_MODEL = "voyage-multimodal-3"  # Voyage的多模态embedding模型
EMBEDDING_DIM = 1024  # voyage-multimodal-3的维度
EMBEDDING_BACKEND = "voyage"  # 使用voyage backend

# Qdrant配置
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "robot_task_records"


set_tracing_disabled(True)
set_default_openai_api("chat_completions")
set_default_openai_client(AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY))

# ===== ② 全局实例 =====
rag_retriever = None


# ===== ③ 初始化函数 =====
def init_rag_system():
    """
    初始化RAG检索系统
    - 连接向量数据库
    - 自动检测并插入数据（如果数据库为空）
    - 初始化Embedding生成器
    - 创建RAG检索器
    """
    global rag_retriever
    
    # 1. 创建向量数据库连接
    vector_db = VectorDB(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
        embedding_dim=EMBEDDING_DIM
    )
    
    # 2. 创建embedding生成器
    embedding_gen = MultimodalEmbedding(
        api_key=VOYAGE_API_KEY,
        model=VOYAGE_MODEL,
        embedding_dim=EMBEDDING_DIM,
        backend=EMBEDDING_BACKEND
    )
    
    # 3. 检查数据库状态，自动插入数据
    record_count = vector_db.get_count()
    print(f"[Info] 数据库中有 {record_count} 条记录")
    
    if record_count == 0:
        print(f"[Info] 数据库为空，开始自动插入数据（来自 robot_records_data.py）")
        print(f"[Info] 共 {len(RECORDS)} 条记录，由于 Voyage API 限速，需要约2-3分钟...")
        
        # 使用 RAGRetriever.insert_text_records() 插入
        rag_retriever_temp = RAGRetriever(vector_db, embedding_gen)
        rag_retriever_temp.insert_text_records(RECORDS)
        
        print(f"[Info] ✓ 数据插入完成，当前共 {vector_db.get_count()} 条记录")
    
    # 4. 创建RAG检索器
    rag_retriever = RAGRetriever(vector_db, embedding_gen)
    
    print("[Info] RAG检索系统初始化完成")


# ===== ④ 检索工具（支持多模态）=====
@function_tool
def search_records_multimodal(
    query_text: str,
    query_image_path: Optional[str] = None,
    top_k: int = 3
) -> str:
    """
    多模态向量检索：在向量数据库中检索与查询最相关的记录
    
    Args:
        query_text: 查询文本（必需）
        query_image_path: 查询图片路径（可选）
        top_k: 返回结果数量
    
    Returns:
        JSON格式的检索结果
    """
    if rag_retriever is None:
        return json.dumps({"error": "RAG系统未初始化"}, ensure_ascii=False)
    
    try:
        results = rag_retriever.search(
            query_text=query_text,
            query_image_path=query_image_path,
            top_k=top_k
        )
        
        return rag_retriever.format_results(results)
    
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ===== ⑤ Retriever Agent（使用向量检索）=====
retriever = Agent(
    name="Retriever",
    instructions=(
        "你是检索员，使用多模态向量检索系统：\n"
        "1) 调用 search_records_multimodal(query_text, query_image_path, top_k=3)\n"
        "   - query_text: 用户问题的文本描述\n"
        "   - query_image_path: 如果用户提供了图片路径，传入此参数（可选）\n"
        "2) 解析返回的JSON结果，结果已按相似度降序排列\n"
        "3) 分析similarity_score（相似度分数，越高越相关）\n"
        "4) 结合success_rate（成功率）给出推荐：\n"
        "   - 优先推荐相似度高且成功率高的\n"
        "   - 如果相似度都很高，选成功率最高的\n"
        "5) 用中文给出推荐理由，说明为什么选择该checkpoint\n"
        "6) 附上检索到的Top-K结果作为参考"
    ),
    tools=[search_records_multimodal],
    model=MODEL,
)


# ===== ⑥ Router Agent =====
router = Agent(
    name="Router",
    instructions=(
        "你是分诊路由：\n"
        "- 如果问题涉及选择模型、ckpt、checkpoint、成功率、记录、检索、推荐策略、任务执行等关键词，"
        "请 handoff 给 Retriever\n"
        "- 如果用户提到了图片或视频，也请 handoff 给 Retriever（支持多模态检索）\n"
        "- 否则用一句中文简要回答"
    ),
    handoffs=[retriever],
    model=MODEL,
)


# ===== ⑦ 测试函数 =====
async def test_multimodal_search():
    """测试多模态检索"""
    print("\n" + "="*60)
    print("测试1: 纯文本检索")
    print("="*60)
    
    query1 = "我想打开笔记本电脑，应该用哪个模型？"
    result1 = await Runner.run(router, input=query1)
    print(f"\n查询: {query1}")
    print(f"回答:\n{result1.final_output}")
    
    print("\n" + "="*60)
    print("测试2: 多模态检索（文本+图片）")
    print("="*60)
    
    # 检查示例图片是否存在
    if os.path.exists(EXAMPLE_IMAGE_PATH):
        query2 = (
            f"根据这张图片 {EXAMPLE_IMAGE_PATH}，"
            "这个场景适合用什么checkpoint？我想完成抓取任务。"
        )
        result2 = await Runner.run(router, input=query2)
        print(f"\n查询: {query2}")
        print(f"回答:\n{result2.final_output}")
    else:
        print(f"[Warning] 示例图片不存在: {EXAMPLE_IMAGE_PATH}")
        print("跳过多模态测试")
    
    print("\n" + "="*60)
    print("测试3: 基于任务类型的检索")
    print("="*60)
    
    query3 = "需要精细操作小物体，推荐什么方案？"
    result3 = await Runner.run(router, input=query3)
    print(f"\n查询: {query3}")
    print(f"回答:\n{result3.final_output}")


# ===== ⑧ 主函数 =====
async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='多模态RAG检索系统 - 需要同时提供文本和图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 运行测试
  python ma_router_retriever_multimodal_qdrant.py
  
  # 多模态查询（必须同时提供--query和--image）
  python ma_router_retriever_multimodal_qdrant.py \\
    --query "根据当前场景推荐checkpoint" \\
    --image "/path/to/scene.jpg"
  
注意:
  - 首次运行会自动插入数据（约2-3分钟）
  - 查询时需要提供文本和图片路径
        '''
    )
    parser.add_argument('--query', type=str, help='查询文本（必须与--image一起使用）')
    parser.add_argument('--image', type=str, help='查询图片路径（必须与--query一起使用）')
    
    args = parser.parse_args()
    
    # 检查参数有效性
    if args.query and not args.image:
        print("❌ 错误: 提供了 --query 但缺少 --image")
        print("💡 提示: 必须同时提供 --query 和 --image 参数")
        print("\n示例:")
        print('  python ma_router_retriever_multimodal_qdrant.py --query "推荐checkpoint" --image "scene.jpg"')
        return
    
    if args.image and not args.query:
        print("❌ 错误: 提供了 --image 但缺少 --query")
        print("💡 提示: 必须同时提供 --query 和 --image 参数")
        print("\n示例:")
        print('  python ma_router_retriever_multimodal_qdrant.py --query "推荐checkpoint" --image "scene.jpg"')
        return
    
    # 如果提供了图片路径，检查文件是否存在
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ 错误: 图片文件不存在: {args.image}")
            return
        print(f"✓ 图片文件存在: {args.image}")
    
    # 初始化RAG系统
    print("\n正在初始化RAG检索系统...")
    print(f"使用 Voyage AI {VOYAGE_MODEL} 模型")
    print(f"向量维度: {EMBEDDING_DIM}")
    
    init_rag_system()
    
    # 执行查询或测试
    if args.query and args.image:
        # 多模态查询
        print("\n" + "="*70)
        print("多模态查询")
        print("="*70)
        print(f"查询文本: {args.query}")
        print(f"查询图片: {args.image}")
        print()
        
        # 构建完整的查询提示
        full_query = f"根据图片 {args.image}，{args.query}"
        result = await Runner.run(router, input=full_query)
        print(result.final_output)
    else:
        # 运行测试
        print("\n💡 未提供查询参数，运行测试模式...")
        await test_multimodal_search()


if __name__ == "__main__":
    asyncio.run(main())

