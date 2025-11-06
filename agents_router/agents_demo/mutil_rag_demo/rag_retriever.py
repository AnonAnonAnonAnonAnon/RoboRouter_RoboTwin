# -*- coding: utf-8 -*-
"""
RAG检索模块
- 多模态Embedding生成（支持文本+图片）
- 语义检索功能
- 支持多种embedding后端（Jina API、本地CLIP等）
"""

import os
import base64
import mimetypes
from typing import List, Dict, Optional
import requests


class MultimodalEmbedding:
    """多模态Embedding生成器"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-multimodal-3",
        embedding_dim: int = 1024,
        backend: str = "voyage"
    ):
        """
        初始化Embedding生成器
        
        Args:
            api_key: API密钥
            model: 模型名称
            embedding_dim: 向量维度
            backend: 使用的后端 ("voyage", "jina", "clip")
        """
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY") or os.environ.get("JINA_API_KEY", "")
        self.model = model
        self.embedding_dim = embedding_dim
        self.backend = backend.lower()
        
        # 根据backend设置
        if self.backend == "voyage":
            self.api_url = None  # Voyage使用SDK
            # 延迟导入voyageai
            self._voyage_client = None
        elif self.backend == "jina":
            self.api_url = "https://api.jina.ai/v1/embeddings"
        
        # 缓存本地CLIP模型（如果使用）
        self._clip_model = None
        self._clip_processor = None
    
    def get_embedding(
        self,
        text: str,
        image_path: Optional[str] = None
    ) -> List[float]:
        """
        获取多模态embedding
        
        Args:
            text: 文本输入
            image_path: 图片路径（可选）
        
        Returns:
            embedding向量
        """
        if self.backend == "voyage":
            return self._get_voyage_embedding(text, image_path)
        elif self.backend == "jina":
            return self._get_jina_embedding(text, image_path)
        else:  # clip
            return self._get_clip_embedding(text, image_path)
    
    def _get_voyage_embedding(
        self,
        text: str,
        image_path: Optional[str] = None
    ) -> List[float]:
        """使用Voyage AI的多模态embedding API"""
        try:
            # 延迟导入voyageai和PIL
            if self._voyage_client is None:
                import voyageai
                self._voyage_client = voyageai.Client(api_key=self.api_key)
                print(f"[Info] 初始化Voyage AI客户端")
            
            # 添加延迟避免超过速率限制（3 RPM = 20秒/次）
            import time
            time.sleep(21)  # 等待21秒，确保不超过3次/分钟
            
            from PIL import Image
            
            # 构建输入：[文本, 图片]的列表
            input_item = [text]
            
            # 如果有图片，添加到输入
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    input_item.append(img)
                    print(f"[Debug] 加载图片: {image_path}")
                except Exception as e:
                    print(f"[Warning] 图片加载失败: {e}，仅使用文本")
            
            # Voyage API接受输入格式：[[text, image], ...]
            inputs = [input_item]
            
            # 调用API
            result = self._voyage_client.multimodal_embed(
                inputs=inputs,
                model=self.model
            )
            
            # 返回第一个结果的embedding
            return result.embeddings[0]
        
        except Exception as e:
            print(f"[Error] Voyage API调用失败: {e}")
            raise RuntimeError(f"Voyage AI embedding生成失败: {e}") from e
    
    def _get_jina_embedding(
        self,
        text: str,
        image_path: Optional[str] = None
    ) -> List[float]:
        """使用Jina AI的多模态embedding API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 构建输入
        input_data = []
        
        # 添加文本
        input_data.append({"text": text})
        
        # 添加图片（如果有）
        if image_path and os.path.exists(image_path):
            # 转换为base64
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = "image/jpeg"
            data_url = f"data:{mime_type};base64,{img_b64}"
            input_data.append({"image": data_url})
        
        payload = {
            "model": self.model,
            "input": input_data,
            "encoding_type": "float"
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Jina返回多个embedding，取平均（文本+图片融合）
            embeddings = [item["embedding"] for item in result["data"]]
            if len(embeddings) == 1:
                return embeddings[0]
            else:
                # 多个输入取平均
                import numpy as np
                return np.mean(embeddings, axis=0).tolist()
        
        except Exception as e:
            print(f"[Error] Jina API调用失败: {e}")
            raise RuntimeError(f"Jina AI embedding生成失败: {e}") from e
    
    def _get_clip_embedding(
        self,
        text: str,
        image_path: Optional[str] = None
    ) -> List[float]:
        """使用本地CLIP模型"""
        try:
            # 延迟加载CLIP模型
            if self._clip_model is None:
                from transformers import CLIPProcessor, CLIPModel
                model_name = "openai/clip-vit-base-patch32"
                self._clip_model = CLIPModel.from_pretrained(model_name)
                self._clip_processor = CLIPProcessor.from_pretrained(model_name)
                print(f"[Info] 加载本地CLIP模型: {model_name}")
            
            from PIL import Image
            import torch
            
            # 处理输入
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path)
                inputs = self._clip_processor(
                    text=[text],
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                # 获取融合后的embedding
                with torch.no_grad():
                    outputs = self._clip_model(**inputs)
                    # 取text和image特征的平均
                    text_embeds = outputs.text_embeds
                    image_embeds = outputs.image_embeds
                    combined = (text_embeds + image_embeds) / 2
                    return combined[0].cpu().numpy().tolist()
            else:
                # 只有文本
                inputs = self._clip_processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True
                )
                with torch.no_grad():
                    outputs = self._clip_model.get_text_features(**inputs)
                    return outputs[0].cpu().numpy().tolist()
        
        except Exception as e:
            print(f"[Error] 本地CLIP模型加载失败: {e}")
            raise RuntimeError(f"本地CLIP embedding生成失败: {e}") from e


class RAGRetriever:
    """RAG检索器：封装向量数据库和embedding"""
    
    def __init__(
        self,
        vector_db,
        embedding_generator: MultimodalEmbedding
    ):
        """
        初始化RAG检索器
        
        Args:
            vector_db: VectorDB实例
            embedding_generator: MultimodalEmbedding实例
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
    
    def insert_text_records(self, records: List[Dict]):
        """
        插入文本记录（为每条记录生成embedding）
        
        Args:
            records: 记录列表，每条记录需包含id和text字段
        """
        def embedding_func(rec):
            # 构建用于embedding的文本
            text = rec.get("text") or rec.get("description") or ""
            if not text:
                # 如果没有text字段，自动拼接其他字段
                text_parts = []
                for key in ["task", "ckpt", "notes", "description"]:
                    if key in rec and rec[key]:
                        text_parts.append(f"{key}: {rec[key]}")
                text = "\n".join(text_parts)
            
            return self.embedding_generator.get_embedding(text)
        
        self.vector_db.insert_records(records, embedding_func)
    
    def search(
        self,
        query_text: str,
        query_image_path: Optional[str] = None,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        多模态检索
        
        Args:
            query_text: 查询文本
            query_image_path: 查询图片路径（可选）
            top_k: 返回top-k结果
            score_threshold: 相似度阈值
        
        Returns:
            检索结果列表
        """
        # 生成查询向量
        query_vector = self.embedding_generator.get_embedding(
            query_text,
            query_image_path
        )
        
        # 向量检索
        results = self.vector_db.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        return results
    
    def format_results(self, results: List[Dict]) -> str:
        """
        格式化检索结果为JSON字符串
        
        Args:
            results: 检索结果列表
        
        Returns:
            JSON格式的字符串
        """
        import json
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "task": r.get("task", ""),
                "ckpt": r.get("ckpt", ""),
                "success_rate": r.get("success_rate", 0.0),
                "notes": r.get("notes", ""),
                "similarity_score": round(r["score"], 4)
            })
        
        return json.dumps(formatted_results, ensure_ascii=False, indent=2)

