# -*- coding: utf-8 -*-
"""
基于 Qwen3-VL 的多模态重排器（Logit-based Batch 方法）
- 支持图片+文本的多模态查询
- 对纯文本文档进行重排
- 使用 Qwen3-VL 视觉语言模型判断相关性
- 采用 Logit-based Batch 评分：一次推理所有候选，从 logits 中提取 yes/no 概率，获得细粒度的连续分数
"""

import torch
from typing import List, Dict, Optional
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os


class Qwen3VLMultimodalReranker:
    """
    基于 Qwen3-VL 的多模态重排器（Logit-based Batch 方法）
    
    特点：
    1. 支持多模态查询（文本 + 图片）
    2. 对文本文档进行重排
    3. 使用 Qwen3-VL 视觉语言模型的判断能力
    4. 采用 Logit-based Batch 评分方法：
       - 将所有候选文档组成 batch，一次性推理
       - 从模型输出的 logits 中提取 yes/no token 的概率
       - 使用 yes 的概率作为相关性分数
       - 得到更细粒度的连续分数（如 0.73, 0.85 等）
       - 只需 1 次推理（而不是 N 次），大幅提升速度
    
    实现原理（类似 Qwen3-Reranker）：
    - 利用 Transformer 的 batch 推理能力
    - 所有候选共享相同的 query image
    - 每个候选对应不同的 document 文本
    - 模型对 batch 中每个样本独立计算 logits
    - 从每个样本的最后一个 token 的 logits 中提取 yes/no 概率
    """
    
    def __init__(
        self,
        model_path: str = "/data/work/public/llm_modles/Qwen3-VL-2B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        batch_size: int = 4
    ):
        """
        初始化 Qwen3-VL 多模态重排器
        
        Args:
            model_path: Qwen3-VL 模型路径（本地路径或 HuggingFace 模型名）
            device: 设备 (cuda/cpu)
            use_fp16: 是否使用半精度（FP16）
            batch_size: 批处理大小
        """
        self.device = device
        self.batch_size = batch_size
        
        print(f"[Qwen3VLReranker] 正在加载 Qwen3-VL 模型: {model_path}")
        print(f"[Qwen3VLReranker] 设备: {device}")
        
        # 使用官方推荐的加载方式
        # 使用 "auto" 让模型自动选择最佳 dtype
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",  # 官方推荐使用 "auto"
            device_map=device,
            trust_remote_code=True
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print(f"[Qwen3VLReranker] 模型加载完成")
        
        # 预处理 yes/no token IDs（用于提取 logits）
        self._init_token_ids()
    
    def _init_token_ids(self):
        """预处理 yes/no token 的 ID"""
        # Qwen3-VL 模型中 yes/no 的表示
        yes_candidates = ["yes", "Yes", "YES", "是"]
        no_candidates = ["no", "No", "NO", "否"]
        
        # 找到 yes token
        self.token_yes_id = None
        for yes_token in yes_candidates:
            token_ids = self.processor.tokenizer.encode(yes_token, add_special_tokens=False)
            if token_ids:
                self.token_yes_id = token_ids[0]
                print(f"[Qwen3VLReranker] Yes token: '{yes_token}' -> ID: {self.token_yes_id}")
                break
        
        # 找到 no token
        self.token_no_id = None
        for no_token in no_candidates:
            token_ids = self.processor.tokenizer.encode(no_token, add_special_tokens=False)
            if token_ids:
                self.token_no_id = token_ids[0]
                print(f"[Qwen3VLReranker] No token: '{no_token}' -> ID: {self.token_no_id}")
                break
        
        if self.token_yes_id is None or self.token_no_id is None:
            print("[Warning] 无法找到 Yes/No token，将使用生成式评分")
            self.use_logit_scoring = False
        else:
            self.use_logit_scoring = True
    
    def format_prompt(
        self,
        query_text: str,
        query_image_path: Optional[str],
        document: str
    ) -> List[Dict]:
        """
        格式化提示词（Qwen3-VL 格式）
        
        Args:
            query_text: 查询文本
            query_image_path: 查询图片路径（可选）
            document: 文档文本
            
        Returns:
            Qwen3-VL 消息格式
        """
        # 构建用户输入内容
        content = []
        
        # 添加图片（如果有）
        if query_image_path and os.path.exists(query_image_path):
            content.append({
                "type": "image",
                "image": query_image_path
            })
        
        # 添加文本提示
        text_prompt = (
            f"Based on the image (if provided) and the query, "
            f"determine if the following document is relevant and suitable.\n\n"
            f"Query: {query_text}\n\n"
            f"Document: {document}\n\n"
            f"Is this document suitable? Answer only 'yes' or 'no'."
        )
        
        content.append({
            "type": "text",
            "text": text_prompt
        })
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
    @torch.no_grad()
    def compute_scores(
        self,
        query_text: str,
        documents: List[str],
        query_image_path: Optional[str] = None
    ) -> List[float]:
        """
        计算查询与文档的相关性分数（Logit-based 批处理版本）
        
        对多个文档进行 batch 推理，从模型的 logits 中提取 yes/no 的概率作为分数。
        这种方法可以得到更细粒度的连续分数，且只需要一次推理（而不是 N 次）。
        
        Args:
            query_text: 查询文本
            documents: 文档列表
            query_image_path: 查询图片路径（可选）
            
        Returns:
            相关性分数列表（0-1之间，越高越相关）
            
        Raises:
            RuntimeError: 如果重排失败且无法恢复
        """
        if not documents:
            return []
        
        try:
            # Batch + Logit-based: 一次性推理所有文档，提取 logits
            print(f"[Qwen3VLReranker] 使用 Logit-based Batch 方法一次性评估 {len(documents)} 个候选")
            
            # 构建 batch messages（所有文档共享相同的 query image）
            batch_messages = []
            for doc in documents:
                messages = self.format_prompt(query_text, query_image_path, doc)
                batch_messages.append(messages)
            
            # 处理 batch inputs
            # 注意：需要逐个处理 messages，然后合并成 batch
            all_input_ids = []
            all_attention_mask = []
            all_pixel_values = []
            all_image_grid_thw = []
            
            for messages in batch_messages:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                all_input_ids.append(inputs['input_ids'])
                all_attention_mask.append(inputs['attention_mask'])
                if 'pixel_values' in inputs:
                    all_pixel_values.append(inputs['pixel_values'])
                if 'image_grid_thw' in inputs:
                    all_image_grid_thw.append(inputs['image_grid_thw'])
            
            # 合并成 batch（需要 padding 到相同长度）
            # 使用 tokenizer 的 pad 功能
            max_len = max(ids.shape[1] for ids in all_input_ids)
            padded_input_ids = []
            padded_attention_mask = []
            
            for input_ids, attention_mask in zip(all_input_ids, all_attention_mask):
                pad_len = max_len - input_ids.shape[1]
                if pad_len > 0:
                    # 左侧 padding（因为我们关心最后一个 token）
                    input_ids = torch.nn.functional.pad(
                        input_ids, (pad_len, 0), value=self.processor.tokenizer.pad_token_id
                    )
                    attention_mask = torch.nn.functional.pad(
                        attention_mask, (pad_len, 0), value=0
                    )
                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attention_mask)
            
            # 组装 batch inputs
            batch_inputs = {
                'input_ids': torch.cat(padded_input_ids, dim=0).to(self.model.device),
                'attention_mask': torch.cat(padded_attention_mask, dim=0).to(self.model.device)
            }
            
            # 如果有图片，也需要 batch
            if all_pixel_values:
                batch_inputs['pixel_values'] = torch.cat(all_pixel_values, dim=0).to(self.model.device)
            if all_image_grid_thw:
                batch_inputs['image_grid_thw'] = torch.cat(all_image_grid_thw, dim=0).to(self.model.device)
            
            print(f"[Qwen3VLReranker] Batch 输入准备完成，shape: {batch_inputs['input_ids'].shape}")
            
            # 前向推理，获取 batch logits
            outputs = self.model(**batch_inputs)
            logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
            
            # 取每个样本最后一个有效 token 的 logits
            # 需要根据 attention_mask 找到最后一个有效位置
            batch_size = logits.shape[0]
            last_token_logits = []
            
            for i in range(batch_size):
                # 找到这个样本的最后一个有效 token 位置
                valid_length = batch_inputs['attention_mask'][i].sum().item()
                last_pos = valid_length - 1
                last_token_logits.append(logits[i, last_pos, :])
            
            last_token_logits = torch.stack(last_token_logits)  # shape: (batch_size, vocab_size)
            
            # 提取 yes/no token 的 logits
            if self.use_logit_scoring:
                yes_logits = last_token_logits[:, self.token_yes_id]  # shape: (batch_size,)
                no_logits = last_token_logits[:, self.token_no_id]    # shape: (batch_size,)
                
                # 使用 softmax 将 logits 转换为概率
                import torch.nn.functional as F
                yes_no_logits = torch.stack([yes_logits, no_logits], dim=1)  # shape: (batch_size, 2)
                yes_no_probs = F.softmax(yes_no_logits, dim=1)  # shape: (batch_size, 2)
                
                # yes 的概率作为相关性分数
                scores = yes_no_probs[:, 0].cpu().tolist()
                
                print(f"[Qwen3VLReranker] ✓ Batch Logit-based 评估完成")
                for i, score in enumerate(scores):
                    print(f"[Qwen3VLReranker]   候选 {i+1}: yes_logit={yes_logits[i].item():.4f}, "
                          f"no_logit={no_logits[i].item():.4f}, score={score:.4f}")
            else:
                # 如果无法使用 logit scoring，回退到生成式评分
                print(f"[Qwen3VLReranker]   ⚠️ logit scoring 不可用，需要使用生成式评分")
                raise RuntimeError("Logit scoring 不可用")
            
            return scores
            
        except Exception as e:
            # 推理失败
            print(f"[Error] Batch Logit-based 推理失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Batch Logit-based 推理失败: {e}") from e
    
    def _score_by_generation(self, inputs) -> float:
        """
        通过生成式方法获取分数（回退方案）
        
        Args:
            inputs: 已经准备好的模型输入
            
        Returns:
            相关性分数（0-1之间）
        """
        # 生成回答
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.1
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip().lower()
        
        # 简单的 yes/no 解析
        if "yes" in output_text or "是" in output_text:
            return 0.95
        else:
            return 0.05
    
    def _format_listwise_prompt(
        self,
        query_text: str,
        query_image_path: Optional[str],
        documents: List[str]
    ) -> List[Dict]:
        """
        格式化 Listwise Reranking 的提示词
        
        Args:
            query_text: 任务描述
            query_image_path: 场景图片路径
            documents: 候选文档列表
            
        Returns:
            Qwen3-VL 消息格式
        """
        # 构建用户输入内容
        content = []
        
        # 添加场景图片（如果有）
        if query_image_path and os.path.exists(query_image_path):
            content.append({
                "type": "image",
                "image": query_image_path
            })
        
        # 构建候选列表
        candidates_text = "\n".join([
            f"{i+1}. {doc}" for i, doc in enumerate(documents)
        ])
        
        # 添加文本提示（让模型打分，0-100分）
        text_prompt = (
            f"You are an expert in robotic manipulation task recommendation.\n\n"
            f"Task: Based on the scene image (if provided) and task description, "
            f"evaluate ALL the following checkpoints and rate their suitability with a score from 0-100.\n\n"
            f"Task Description: {query_text}\n\n"
            f"Candidate Checkpoints:\n{candidates_text}\n\n"
            f"Instructions:\n"
            f"- For EACH checkpoint (1 to {len(documents)}), give a suitability score from 0 to 100\n"
            f"- 100 = perfect match for this task\n"
            f"- 80-99 = highly suitable\n"
            f"- 60-79 = suitable\n"
            f"- 40-59 = moderately suitable\n"
            f"- 20-39 = less suitable\n"
            f"- 0-19 = not suitable\n"
            f"- Consider the scene, task requirements, checkpoint capabilities, and success rates\n"
            f"- Be critical and give different scores to distinguish quality\n\n"
            f"Output format (one per line, MUST evaluate all {len(documents)} checkpoints):\n"
            f"1: <score>\n"
            f"2: <score>\n"
            f"...\n"
            f"{len(documents)}: <score>\n\n"
            f"Your evaluation:"
        )
        
        content.append({
            "type": "text",
            "text": text_prompt
        })
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
    def _parse_listwise_output(self, output_text: str, num_docs: int) -> List[float]:
        """
        解析 Listwise Reranking 的输出（支持 0-100 分数）
        
        Args:
            output_text: 模型输出文本
            num_docs: 文档数量
            
        Returns:
            分数列表（归一化到 0-1）
            
        Raises:
            ValueError: 如果有候选未被评估
        """
        import re
        
        scores = []
        unevaluated = []  # 记录未被评估的候选
        
        # 按行解析
        lines = output_text.strip().split('\n')
        
        for i in range(num_docs):
            score = None  # 初始为 None，表示未评估
            
            # 寻找对应编号的评估
            for line in lines:
                line_stripped = line.strip()
                
                # 匹配格式：1: 85 或 1. 85 或 1: score=85
                # 支持多种格式
                patterns = [
                    rf"^{i+1}:\s*(\d+\.?\d*)",  # 1: 85
                    rf"^{i+1}\.\s*(\d+\.?\d*)",  # 1. 85
                    rf"^{i+1}:\s*score\s*[=:]\s*(\d+\.?\d*)",  # 1: score=85
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, line_stripped, re.IGNORECASE)
                    if match:
                        try:
                            raw_score = float(match.group(1))
                            # 归一化到 0-1 范围（假设输入是 0-100）
                            if raw_score > 1.0:  # 看起来是 0-100 的分数
                                score = raw_score / 100.0
                            else:  # 已经是 0-1 的分数
                                score = raw_score
                            
                            # 确保在 0-1 范围内
                            score = max(0.0, min(1.0, score))
                            break
                        except ValueError:
                            continue
                
                if score is not None:
                    break
            
            # 如果没有找到评估，记录下来
            if score is None:
                unevaluated.append(i+1)
                score = 0.01  # 临时给个极低分
            
            scores.append(score)
        
        # 检查是否所有候选都被评估了
        if unevaluated:
            error_msg = (
                f"❌ Listwise Reranking 失败：有 {len(unevaluated)} 个候选未被评估: {unevaluated}\n"
                f"模型输出:\n{output_text}\n\n"
                f"可能的原因:\n"
                f"1. max_new_tokens 太小，生成被截断\n"
                f"2. prompt 格式不清晰，模型没理解\n"
                f"3. 模型输出格式不规范\n\n"
                f"建议:\n"
                f"1. 增加 max_new_tokens（当前为200）\n"
                f"2. 检查模型输出格式\n"
                f"3. 优化 prompt 设计"
            )
            print(f"[Error] {error_msg}")
            raise ValueError(error_msg)
        
        return scores
    
    def rerank(
        self,
        query_text: str,
        results: List[Dict],
        query_image_path: Optional[str] = None,
        top_k: Optional[int] = None,
        doc_field: str = "text"
    ) -> List[Dict]:
        """
        对检索结果进行重排
        
        Args:
            query_text: 查询文本
            results: 初始检索结果列表（每项为字典）
            query_image_path: 查询图片路径（可选）
            top_k: 返回前k个结果（None表示返回全部）
            doc_field: 用作文档内容的字段名
            
        Returns:
            重排后的结果列表（添加了 rerank_score 字段）
        """
        if not results:
            return []
        
        # 提取文档文本
        documents = []
        for r in results:
            # 支持多字段组合
            if doc_field in r:
                doc_text = r[doc_field]
            else:
                # 如果没有指定字段，尝试组合多个字段
                doc_parts = []
                for field in ["task", "ckpt", "notes", "description", "text"]:
                    if field in r and r[field]:
                        doc_parts.append(f"{field}: {r[field]}")
                doc_text = "\n".join(doc_parts) if doc_parts else str(r)
            
            documents.append(doc_text)
        
        # 计算重排分数
        rerank_scores = self.compute_scores(
            query_text=query_text,
            documents=documents,
            query_image_path=query_image_path
        )
        
        # 将分数添加到结果中
        for i, result in enumerate(results):
            result["rerank_score"] = rerank_scores[i]
        
        # 按重排分数降序排序
        reranked_results = sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        
        # 返回 top-k
        if top_k is not None:
            reranked_results = reranked_results[:top_k]
        
        return reranked_results


class TaskRecommendationQwen3VLReranker(Qwen3VLMultimodalReranker):
    """
    针对任务推荐场景的 Qwen3-VL 多模态重排器
    
    专门优化提示词，用于判断：
    "给定任务场景（图片+描述），该checkpoint/模型是否适合完成任务？"
    """
    
    def format_prompt(
        self,
        query_text: str,
        query_image_path: Optional[str],
        document: str
    ) -> List[Dict]:
        """
        为任务推荐场景格式化提示词
        
        Args:
            query_text: 任务描述
            query_image_path: 场景图片路径
            document: checkpoint/模型信息
            
        Returns:
            Qwen3-VL 消息格式
        """
        # 构建用户输入内容
        content = []
        
        # 添加场景图片（如果有）
        if query_image_path and os.path.exists(query_image_path):
            content.append({
                "type": "image",
                "image": query_image_path
            })
        
        # 添加文本提示（针对机器人任务推荐优化）
        text_prompt = (
            f"You are an expert in robotic manipulation task recommendation.\n\n"
            f"Given the scene image (if provided) and task description, "
            f"evaluate whether the following checkpoint/model is suitable for this task.\n\n"
            f"Task Description: {query_text}\n\n"
            f"Checkpoint/Model Information:\n{document}\n\n"
            f"Based on the scene and task requirements, is this checkpoint suitable? "
            f"Answer only 'yes' or 'no'."
        )
        
        content.append({
            "type": "text",
            "text": text_prompt
        })
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages


# 便捷函数：创建默认的多模态重排器
def create_qwen3vl_reranker(
    model_path: str = "/data/work/public/llm_modles/Qwen3-VL-2B-Instruct",
    device: str = "auto",
    task_specific: bool = True
) -> Qwen3VLMultimodalReranker:
    """
    创建 Qwen3-VL 多模态重排器实例
    
    Args:
        model_path: Qwen3-VL 模型路径
        device: 设备（auto/cuda/cpu）
        task_specific: 是否使用任务推荐专用的重排器
        
    Returns:
        Qwen3-VL 多模态重排器实例
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if task_specific:
        return TaskRecommendationQwen3VLReranker(
            model_path=model_path,
            device=device
        )
    else:
        return Qwen3VLMultimodalReranker(
            model_path=model_path,
            device=device
        )


if __name__ == "__main__":
    # 测试代码
    print("="*70)
    print("测试 Qwen3-VL 多模态重排器")
    print("="*70)
    
    # 使用本地模型路径
    MODEL_PATH = "/data/work/public/llm_modles/Qwen3-VL-2B-Instruct"
    
    # 创建重排器
    reranker = TaskRecommendationQwen3VLReranker(model_path=MODEL_PATH)
    
    # 测试数据
    query_text = "我需要抓取桌面上的笔记本电脑"
    example_image = "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/frames_to_push/f_0.jpg"
    
    # 模拟检索结果
    results = [
        {
            "task": "拾取钢笔",
            "ckpt": "pen_pickup_v2.pth",
            "notes": "适用于小物体精细抓取",
            "success_rate": 0.88,
            "similarity_score": 0.85
        },
        {
            "task": "抓取笔记本电脑",
            "ckpt": "laptop_grasp_v1.pth",
            "notes": "专门用于笔记本电脑抓取",
            "success_rate": 0.92,
            "similarity_score": 0.80
        },
        {
            "task": "通用物体抓取",
            "ckpt": "general_grasp.pth",
            "notes": "通用抓取模型，适用于多种物体",
            "success_rate": 0.75,
            "similarity_score": 0.78
        },
    ]
    
    print(f"\n查询: {query_text}")
    if os.path.exists(example_image):
        print(f"图片: {example_image}\n")
    else:
        print("图片: 无（纯文本模式）\n")
    
    print(f"原始检索结果（按向量相似度排序）:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['task']} (相似度: {r['similarity_score']:.4f}, 成功率: {r['success_rate']:.2f})")
    
    # 重排
    print("\n正在进行多模态重排...")
    reranked = reranker.rerank(
        query_text=query_text,
        results=results,
        query_image_path=example_image if os.path.exists(example_image) else None,
        top_k=3
    )
    
    print(f"\n重排后的结果（按重排分数排序）:")
    for i, r in enumerate(reranked, 1):
        print(f"{i}. {r['task']} (重排分数: {r['rerank_score']:.4f}, 成功率: {r['success_rate']:.2f})")

