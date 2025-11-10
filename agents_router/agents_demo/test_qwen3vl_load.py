#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Qwen3-VL 模型加载
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

print("=" * 70)
print("测试 Qwen3-VL 模型加载")
print("=" * 70)

try:
    print("\n[1/4] 导入 transformers...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print("✓ 导入成功")
    
    print("\n[2/4] 检查 transformers 版本...")
    import transformers
    print(f"✓ transformers 版本: {transformers.__version__}")
    
    print("\n[3/4] 加载模型...")
    model_path = "/data/work/public/llm_modles/Qwen3-VL-2B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="cuda",
        trust_remote_code=True
    )
    print(f"✓ 模型加载成功！模型类型: {type(model).__name__}")
    
    print("\n[4/4] 加载 Processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"✓ Processor 加载成功")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！Qwen3-VL 可以正常使用")
    print("=" * 70)
    
except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    print("\n可能的解决方案:")
    print("1. 升级 transformers: pip install --upgrade transformers")
    print("2. 安装最新版本: pip install transformers>=4.50.0")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n可能的解决方案:")
    print("1. 检查模型路径是否正确")
    print("2. 确保有足够的 GPU 内存")
    print("3. 升级 transformers 到最新版本")

