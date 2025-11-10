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
        embedding_generator: MultimodalEmbedding,
        reranker=None,
        use_reranker: bool = True
    ):
        """
        初始化RAG检索器
        
        Args:
            vector_db: VectorDB实例
            embedding_generator: MultimodalEmbedding实例
            reranker: 重排器实例（可选，如果为None且use_reranker=True则延迟加载）
            use_reranker: 是否使用重排器
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.reranker = reranker
        self.use_reranker = use_reranker
        self._reranker_initialized = False
    
    def _init_reranker(self):
        """
        延迟初始化重排器（已弃用）
        
        注意：重排器应该在主程序中显式初始化并传入。
        这个方法只是一个保护性检查，确保重排器已经被正确初始化。
        """
        if not self._reranker_initialized:
            if self.reranker is None and self.use_reranker:
                print("[RAGRetriever] ⚠️ 重排器未初始化！")
                print("[RAGRetriever] 请在主程序中初始化重排器并传入 RAGRetriever")
                print("[RAGRetriever] 将禁用重排功能")
                self.use_reranker = False
            self._reranker_initialized = True
    
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
        score_threshold: Optional[float] = None,
        rerank_top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        多模态检索（支持重排）
        
        工作流程：
        1. 第一步：向量检索（召回阶段）
           - 使用 embedding 生成查询向量（多模态：图片+文本）
           - 在向量数据库中检索相似的候选结果
           - 如果启用重排，召回数量会是 top_k * 2，为重排提供更多候选
        
        2. 第二步：重排（精排阶段）
           - 对召回的候选结果进行精确判断
           - 使用 Qwen3-VL 视觉语言模型同时理解图片和文本
           - 判断每个候选是否真正适合完成任务
           - 按重排分数排序，返回最终结果
        
        Args:
            query_text: 查询文本
            query_image_path: 查询图片路径（可选）
            top_k: 向量检索返回的结果数量
            score_threshold: 相似度阈值
            rerank_top_k: 重排后返回的结果数量（None表示返回全部重排结果）
        
        Returns:
            检索结果列表（如果启用重排，则按重排分数排序）
        """
        # 第一步：向量检索（召回阶段）
        # 如果启用重排，可以召回更多候选，例如 top_k * 2
        initial_top_k = top_k * 2 if self.use_reranker else top_k
        
        print(f"[RAGRetriever] 第1步：向量检索，召回 top-{initial_top_k} 候选")
        
        # 生成查询向量
        query_vector = self.embedding_generator.get_embedding(
            query_text,
            query_image_path
        )
        
        # 向量检索
        results = self.vector_db.search(
            query_vector=query_vector,
            top_k=initial_top_k,
            score_threshold=score_threshold
        )
        
        if not results:
            return []
        
        # 第二步：重排（精排阶段）
        if self.use_reranker:
            print(f"[RAGRetriever] 第2步：多模态重排（Qwen3-VL）")
            # 延迟初始化重排器
            self._init_reranker()
            
            if self.reranker is not None:
                try:
                    print(f"[RAGRetriever] 对 {len(results)} 个候选结果进行重排...")
                    
                    # 检查重排器是否支持多模态（是否有 query_image_path 参数）
                    import inspect
                    rerank_signature = inspect.signature(self.reranker.rerank)
                    supports_multimodal = 'query_image_path' in rerank_signature.parameters
                    
                    # 执行重排
                    if supports_multimodal:
                        # 多模态重排器（如 Qwen3-VL）
                        results = self.reranker.rerank(
                            query_text=query_text,
                            results=results,
                            query_image_path=query_image_path,
                            top_k=rerank_top_k or top_k  # 使用指定的重排top_k或默认top_k
                        )
                    else:
                        # 传统文本重排器
                        results = self.reranker.rerank(
                            query=query_text,
                            results=results,
                            top_k=rerank_top_k or top_k
                        )
                    
                    print(f"[RAGRetriever] 重排完成，返回 top-{len(results)} 结果")
                except Exception as e:
                    print(f"[Warning] 重排失败: {e}")
                    print("[Warning] 返回原始检索结果")
                    # 如果重排失败，截取原始结果的 top_k
                    results = results[:top_k]
            else:
                # 重排器未初始化，截取 top_k
                results = results[:top_k]
        
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
            result_dict = {
                "task": r.get("task", ""),
                "ckpt": r.get("ckpt", ""),
                "success_rate": r.get("success_rate", 0.0),
                "notes": r.get("notes", ""),
                "similarity_score": round(r["score"], 4)
            }
            
            # 如果有重排分数，也加入结果
            if "rerank_score" in r:
                result_dict["rerank_score"] = round(r["rerank_score"], 4)
            
            formatted_results.append(result_dict)
        
        return json.dumps(formatted_results, ensure_ascii=False, indent=2)

