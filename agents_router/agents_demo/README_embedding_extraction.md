# 多模态Embedding提取和t-SNE可视化 - 使用说明

## 📋 概述

本工具用于从RoboTwin视频数据中提取多模态embedding，并使用t-SNE进行可视化，以分析不同机器人任务的表征聚类效果。

## 🗂️ 文件说明

### 1. Embedding提取脚本
- **`extract_multimodel_embeding_5task_20data_local.py`** - 使用本地Ops-MM-embedding-v1-2B模型
  - 模型路径: `/data/work/public/llm_modles/Ops-MM-embedding-v1-2B`
  - 使用GPU 7
  - 处理5个任务，每个任务前20个视频
  - 自动fallback到eager attention（如果flash_attention_2不可用）

### 2. 可视化脚本
- **`draw_t_sne_v2.py`** - 生成3个版本的t-SNE可视化图
  - 版本1: 彩色散点 + 任务名称标注
  - 版本2: 彩色散点，无任务名称
  - 版本3: 灰色背景 + 红色高亮指定任务

## 🚀 使用步骤

### 步骤1: 提取Embeddings

```bash
# 确保在robo conda环境中
conda activate robo

# 运行提取脚本（使用GPU 7）
cd /data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/agents_demo
python extract_multimodel_embeding_5task_20data_local.py
```

**预期输出:**
- 首帧图片保存到: `mm_embed_cache/frames/`
- Embedding JSON保存到: `mm_embed_cache/embeddings/`
- Manifest JSON保存到: `mm_embed_cache/manifest_*.json`

**处理信息:**
- 5个任务 × 20个视频 = 100个样本
- 每个样本约需5-10秒（取决于模型推理速度）
- 总耗时约8-15分钟

### 步骤2: 生成t-SNE可视化

#### 2.1 生成所有3个版本的图

```bash
python draw_t_sne_v2.py
```

#### 2.2 只生成特定版本

```bash
# 只生成版本1（带标注）
python draw_t_sne_v2.py --version 1

# 只生成版本2（无标注）
python draw_t_sne_v2.py --version 2

# 只生成版本3（灰色+高亮）
python draw_t_sne_v2.py --version 3
```

#### 2.3 版本3高亮指定任务

```bash
# 高亮 adjust_bottle 和 open_laptop 任务
python draw_t_sne_v2.py --version 3 --highlight adjust_bottle open_laptop
```

**输出文件:**
- `mm_embed_cache/figures/tsne_TIMESTAMP_v1_with_labels.png` - 版本1
- `mm_embed_cache/figures/tsne_TIMESTAMP_v2_no_labels.png` - 版本2
- `mm_embed_cache/figures/tsne_TIMESTAMP_v3_highlight.png` - 版本3
- `mm_embed_cache/figures/tsne_TIMESTAMP.csv` - 坐标数据
- `mm_embed_cache/figures/tsne_TIMESTAMP.json` - 参数记录

## 📊 当前数据状态

### 视频数据
- **位置**: `/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/video_test/from_dataset/`
- **任务数**: 5个
- **每任务视频数**: 50个 (episode0.mp4 ~ episode49.mp4)

### 5个任务列表
1. `adjust_bottle` - 调整瓶子对齐
2. `beat_block_hammer` - 用锤子敲击方块
3. `click_alarmclock` - 点击闹钟按钮
4. `open_laptop` - 打开笔记本电脑
5. `place_container_plate` - 将容器放到盘子上

## 🛠️ 配置参数

### Embedding提取参数
在 `extract_multimodel_embeding_5task_20data_local.py` 中修改:

```python
VIDEO_PER_TASK = 20  # 每个任务处理的视频数量
EMBEDDING_DIM = 1024  # Embedding维度
```

### t-SNE可视化参数
在 `draw_t_sne_v2.py` 中修改:

```python
USE_PCA = True        # 是否先PCA降维
PCA_DIM = 50          # PCA目标维度
TSNE_PERPLEXITY = 15  # t-SNE perplexity参数
TSNE_N_ITER = 1000    # t-SNE迭代次数
RANDOM_STATE = 42     # 随机种子（保证可复现）
```

## 📈 预期结果

### 好的聚类效果
- 不同任务的点聚成明显的"簇"
- 同任务的点距离较近
- 不同任务的点距离较远

### 解释
- **聚类良好** → Embedding学到了有用的任务特征
- **混成一团** → 模型没有很好地区分不同任务

## 🔄 下一步扩展

### 扩展到50个任务
1. 从RoboTwin2.0下载其他45个任务的视频
2. 修改 `VIDEOS` 字典，添加新任务
3. 重新运行提取和可视化

### 优化建议
- 增加每个任务的样本数（如30-50个）
- 调整t-SNE参数以获得更好的可视化效果
- 使用不同的colormap以适应更多任务

## ❓ 常见问题

**Q: 提取embedding很慢怎么办？**
A: 本地模型推理需要时间，可以考虑:
- 减少 `VIDEO_PER_TASK` 数量
- 使用更强的GPU
- 批量处理多个样本

**Q: t-SNE结果不理想怎么办？**
A: 尝试调整参数:
- 增加 `TSNE_PERPLEXITY` (5-50之间)
- 增加 `TSNE_N_ITER` (1000-5000)
- 调整 `PCA_DIM` (20-100)
- 改变 `RANDOM_STATE` 尝试不同的初始化

**Q: GPU内存不足怎么办？**
A: 
- 确保只使用一个GPU (`CUDA_VISIBLE_DEVICES=7`)
- 关闭其他占用GPU的进程
- 使用eager attention而非flash_attention_2（脚本已自动处理）

## 📝 日志和调试

### 检查embedding提取进度
```bash
# 查看已生成的embedding数量
ls mm_embed_cache/embeddings/*.json | wc -l

# 查看最新的manifest
ls -lt mm_embed_cache/manifest_*.json | head -1
```

### 检查GPU使用情况
```bash
nvidia-smi
# 确认GPU 7正在使用
```

### 查看生成的图片
```bash
ls -lt mm_embed_cache/figures/*.png | head -5
```

## 📚 参考资料

- **RoboTwin2.0数据集**: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0
- **Ops-MM-embedding-v1**: 支持文本、图像、文本-图像对的统一embedding
- **t-SNE**: 高维数据可视化的经典方法

---

**最后更新**: 2025-11-11
**维护者**: OliverRen

