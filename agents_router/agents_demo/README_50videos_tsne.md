# t-SNE可视化 - 50视频版本

## 快速使用

```bash
# 激活环境
conda activate robo

# 生成3个版本的图
python draw_t_sne_50videos.py --version all

# 只生成某个版本
python draw_t_sne_50videos.py --version 1  # 彩色+标注
python draw_t_sne_50videos.py --version 2  # 彩色无标注
python draw_t_sne_50videos.py --version 3  # 灰色+高亮
```

## 参数调整

在 `draw_t_sne_50videos.py` 文件的第20-31行调整：

### 控制图的紧凑度
- **TSNE_PERPLEXITY**（困惑度）：当前120
  - ↑增大 → 图更紧凑
  - ↓减小 → 簇分得更开
  
- **TSNE_EARLY_EXAG**（早期放大）：当前4
  - ↑增大 → 簇之间距离更大
  - ↓减小 → 图更紧凑

### 控制点的大小
- **POINT_SIZE_V1/V2**：当前100
  - 调大 → 点更明显
  - 调小 → 点更精细

## 输出位置

- **图像**: `mm_embed_cache_50videos/figures/`
- **数据**: 1000个样本（20任务 × 50视频）

修改参数后重新运行即可生成新图。

