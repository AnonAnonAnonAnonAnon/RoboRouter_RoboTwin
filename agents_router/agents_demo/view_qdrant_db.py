#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant向量数据库查看工具
- 查看数据库中存储的所有记录
- 显示记录的详细信息
"""

import sys
from mutil_rag_demo.vector_db import VectorDB

# Qdrant配置
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "robot_task_records"
EMBEDDING_DIM = 1024


def list_all_records(vector_db: VectorDB):
    """列出数据库中所有记录的详细信息"""
    record_count = vector_db.get_count()
    
    print("\n" + "="*70)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"数据库中共有 {record_count} 条记录")
    print("="*70)
    
    if record_count == 0:
        print("数据库为空")
        return
    
    # 尝试读取ID 1-50
    found_count = 0
    for i in range(1, 51):
        try:
            record = vector_db.get_by_id(i)
            if record:
                found_count += 1
                print(f"\n记录 ID: {record['id']}")
                print(f"  任务 (task): {record.get('task', 'N/A')}")
                print(f"  检查点 (ckpt): {record.get('ckpt', 'N/A')}")
                print(f"  成功率 (success_rate): {record.get('success_rate', 'N/A')}")
                print(f"  备注 (notes): {record.get('notes', 'N/A')}")
                if 'description' in record:
                    desc = record['description']
                    if len(desc) > 60:
                        print(f"  描述: {desc[:60]}...")
                    else:
                        print(f"  描述: {desc}")
        except Exception as e:
            # ID不存在或其他错误，跳过
            continue
    
    print("\n" + "="*70)
    print(f"共找到 {found_count} 条记录")
    print("="*70)


def main():
    print("Qdrant向量数据库查看工具")
    print(f"连接到: {QDRANT_HOST}:{QDRANT_PORT}")
    
    try:
        # 创建向量数据库连接
        vector_db = VectorDB(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=COLLECTION_NAME,
            embedding_dim=EMBEDDING_DIM
        )
        
        # 列出所有记录
        list_all_records(vector_db)
        
    except Exception as e:
        print(f"\n[Error] 连接数据库失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

