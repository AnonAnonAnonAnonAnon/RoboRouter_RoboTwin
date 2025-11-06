#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤2: 简单检索脚本
功能：输入图片+文字，执行向量检索，返回结果
不使用Agent，只做纯检索
"""

import os
import sys
import json
from typing import Optional

# 导入本地模块
from vector_db import VectorDB
from rag_retriever import MultimodalEmbedding, RAGRetriever


# ===== 配置 =====
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "robot_task_records"

# Embedding配置
EMBEDDING_BACKEND = "voyage"  # "voyage", "jina", "clip"
EMBEDDING_API_KEY = "pa-tEigYTRrvWMOSB2WIcH4f6FkhzEwF8xZ3kUHbFG4hz9"  # Voyage AI API Key
EMBEDDING_MODEL = "voyage-multimodal-3"
EMBEDDING_DIM = 1024


class SimpleSearchService:
    """简单的检索服务"""
    
    def __init__(self):
        """初始化检索服务"""
        print("正在初始化检索服务...")
        
        # 连接向量数据库
        self.vector_db = VectorDB(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=COLLECTION_NAME,
            embedding_dim=EMBEDDING_DIM
        )
        print(f"  ✓ 连接向量数据库: {QDRANT_HOST}:{QDRANT_PORT}")
        
        # 创建embedding生成器
        self.embedding_gen = MultimodalEmbedding(
            api_key=EMBEDDING_API_KEY,
            model=EMBEDDING_MODEL,
            embedding_dim=EMBEDDING_DIM,
            backend=EMBEDDING_BACKEND
        )
        backend_name = {
            "voyage": "Voyage AI",
            "jina": "Jina AI",
            "clip": "本地CLIP"
        }.get(EMBEDDING_BACKEND, EMBEDDING_BACKEND)
        print(f"  ✓ Embedding生成器就绪 ({backend_name})")
        
        # 创建RAG检索器
        self.retriever = RAGRetriever(self.vector_db, self.embedding_gen)
        print(f"  ✓ 检索服务初始化完成")
        
        # 显示数据库状态
        count = self.vector_db.get_count()
        print(f"  ✓ 数据库中有 {count} 条记录可检索")
        print()
    
    def search(
        self,
        text: str,
        image_path: Optional[str] = None,
        top_k: int = 3
    ) -> dict:
        """
        执行检索
        
        Args:
            text: 查询文本
            image_path: 图片路径（可选）
            top_k: 返回结果数量
        
        Returns:
            检索结果字典
        """
        print(f"[查询] {text}")
        if image_path:
            if os.path.exists(image_path):
                print(f"[图片] {image_path}")
            else:
                print(f"[警告] 图片不存在: {image_path}")
                image_path = None
        
        # 执行检索
        results = self.retriever.search(
            query_text=text,
            query_image_path=image_path,
            top_k=top_k
        )
        
        # 整理输出
        output = {
            "query": {
                "text": text,
                "image": image_path,
                "top_k": top_k
            },
            "results": results,
            "count": len(results)
        }
        
        return output
    
    def format_results(self, results: list) -> str:
        """格式化输出结果"""
        if not results:
            return "未找到匹配结果"
        
        output = []
        output.append("\n" + "="*60)
        output.append("  检索结果")
        output.append("="*60)
        
        for i, r in enumerate(results, 1):
            output.append(f"\n[结果 {i}]")
            output.append(f"  任务: {r.get('task', 'N/A')}")
            output.append(f"  Checkpoint: {r.get('ckpt', 'N/A')}")
            output.append(f"  成功率: {r.get('success_rate', 0)*100:.1f}%")
            output.append(f"  相似度: {r.get('score', 0):.4f}")
            output.append(f"  备注: {r.get('notes', 'N/A')}")
            if 'description' in r:
                output.append(f"  描述: {r['description']}")
        
        output.append("\n" + "="*60)
        
        # 推荐最佳结果
        if results:
            best = results[0]
            output.append(f"\n💡 推荐: {best['ckpt']} (成功率{best['success_rate']*100:.1f}%, 相似度{best['score']:.4f})")
        
        return "\n".join(output)
    
    def search_and_print(
        self,
        text: str,
        image_path: Optional[str] = None,
        top_k: int = 3
    ):
        """检索并打印结果"""
        output = self.search(text, image_path, top_k)
        print(self.format_results(output['results']))
        return output


def interactive_mode():
    """交互式检索模式"""
    print("\n" + "="*60)
    print("  交互式检索模式")
    print("="*60)
    print("\n使用说明:")
    print("  - 输入查询文本进行检索")
    print("  - 输入 'image:路径' 可添加图片")
    print("  - 输入 'quit' 或 'exit' 退出")
    print()
    
    service = SimpleSearchService()
    
    while True:
        try:
            query = input("\n请输入查询 > ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            # 检查是否包含图片
            image_path = None
            if 'image:' in query:
                parts = query.split('image:')
                query = parts[0].strip()
                image_path = parts[1].strip()
            
            # 执行检索
            service.search_and_print(query, image_path, top_k=3)
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def batch_test():
    """批量测试"""
    print("\n" + "="*60)
    print("  批量测试模式")
    print("="*60)
    
    service = SimpleSearchService()
    
    test_cases = [
        {
            "text": "打开笔记本电脑",
            "image": None,
            "description": "纯文本检索 - 笔记本任务"
        },
        {
            "text": "抓取小物体",
            "image": None,
            "description": "纯文本检索 - 精细抓取"
        },
        {
            "text": "需要高成功率的方案",
            "image": None,
            "description": "纯文本检索 - 按成功率"
        },
        {
            "text": "敲击木块",
            "image": None,
            "description": "纯文本检索 - 敲击任务"
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n\n{'='*60}")
        print(f"测试 {i}/{len(test_cases)}: {case['description']}")
        print('='*60)
        
        service.search_and_print(
            text=case['text'],
            image_path=case['image'],
            top_k=3
        )


def single_search(text: str, image_path: Optional[str] = None, top_k: int = 3):
    """单次检索（适合脚本调用）"""
    service = SimpleSearchService()
    output = service.search(text, image_path, top_k)
    
    # 打印格式化结果
    print(service.format_results(output['results']))
    
    # 返回JSON（方便程序调用）
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="简单检索服务")
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="查询文本"
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        default=None,
        help="图片路径（可选）"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=3,
        help="返回结果数量（默认3）"
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["single", "interactive", "batch"],
        default="single",
        help="运行模式: single(单次), interactive(交互), batch(批量测试)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出JSON格式"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "interactive":
            # 交互模式
            interactive_mode()
        
        elif args.mode == "batch":
            # 批量测试
            batch_test()
        
        else:
            # 单次检索
            if not args.query:
                print("错误: 请提供查询文本 (-q)")
                print("示例: python 02_simple_search.py -q '打开笔记本电脑'")
                print("或运行交互模式: python 02_simple_search.py -m interactive")
                sys.exit(1)
            
            output = single_search(args.query, args.image, args.top_k)
            
            # 如果需要JSON输出
            if args.json:
                print("\n[JSON输出]")
                print(json.dumps(output, ensure_ascii=False, indent=2))
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

