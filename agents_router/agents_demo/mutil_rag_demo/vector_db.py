# -*- coding: utf-8 -*-
"""
向量数据库管理模块
- 封装Qdrant向量数据库操作
- 支持collection管理、数据插入、向量检索
"""

from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorDB:
    """向量数据库管理类"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "robot_task_records",
        embedding_dim: int = 768,
        use_memory: bool = False
    ):
        """
        初始化Qdrant客户端
        
        Args:
            host: Qdrant服务器地址
            port: Qdrant服务器端口
            collection_name: collection名称
            embedding_dim: 向量维度
            use_memory: 是否使用内存模式（测试用）
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # 初始化客户端
        if use_memory:
            print("[Info] 使用内存模式Qdrant")
            self.client = QdrantClient(":memory:")
        else:
            try:
                self.client = QdrantClient(host=host, port=port)
                print(f"[Info] 连接到Qdrant: {host}:{port}")
            except Exception as e:
                print(f"[Warning] 无法连接到Qdrant服务器，使用内存模式: {e}")
                self.client = QdrantClient(":memory:")
        
        self._init_collection()
    
    def _init_collection(self):
        """初始化collection"""
        # 检查collection是否存在
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name in collection_names:
            print(f"[Info] Collection '{self.collection_name}' 已存在")
            return
        
        # 创建collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        print(f"[Info] 创建Collection '{self.collection_name}'，维度={self.embedding_dim}")
    
    def insert_records(
        self,
        records: List[Dict],
        embedding_func: callable
    ):
        """
        批量插入记录
        
        Args:
            records: 记录列表，每条记录必须包含id和用于生成embedding的数据
            embedding_func: embedding生成函数，接收record，返回向量
        """
        points = []
        for rec in records:
            # 调用embedding函数生成向量
            try:
                embedding = embedding_func(rec)
            except Exception as e:
                print(f"[Error] 生成embedding失败 (record_id={rec.get('id')}): {e}")
                continue
            
            # 构建payload（元数据）
            payload = {k: v for k, v in rec.items() if k != "id"}
            
            point = PointStruct(
                id=rec["id"],
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        # 批量插入
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"[Info] 插入 {len(points)} 条记录到向量数据库")
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        向量检索
        
        Args:
            query_vector: 查询向量
            top_k: 返回top-k结果
            score_threshold: 相似度阈值（可选）
        
        Returns:
            检索结果列表，每个结果包含id、score和payload
        """
        # 向量检索
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        # 格式化结果
        results = []
        for hit in search_result:
            result = {
                "id": hit.id,
                "score": hit.score,
                **hit.payload
            }
            results.append(result)
        
        return results
    
    def delete_collection(self):
        """删除collection"""
        self.client.delete_collection(self.collection_name)
        print(f"[Info] 删除Collection '{self.collection_name}'")
    
    def get_count(self) -> int:
        """获取collection中的记录数量"""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count
    
    def get_by_id(self, record_id: int) -> Optional[Dict]:
        """根据ID获取记录"""
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[record_id]
        )
        if result:
            return {
                "id": result[0].id,
                **result[0].payload
            }
        return None

