# t-SNE可视化 - 50视频版本

## 快速使用

```bash
# 激活环境
conda activate robo
conda activate RoboTwin

# 生成3个版本的图
python draw_t_sne_50videos.py --version all
python /home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/draw_t_sne_50videos_20tasks_draw.py --version all

# 高亮 adjust_bottle 里前 70% 的点，和 shake_bottle 里后 30% 的点
python /home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/draw_t_sne_50videos_20tasks_draw.py --version 3 \
  --highlight adjust_bottle:0.7:head shake_bottle:0.3:tail

#act
python /home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/draw_t_sne_50videos_20tasks_draw.py --version 3 \
  --highlight adjust_bottle:0.9:head lift_pot:0.7:head place_empty_cup:0.6:head

#dp
python /home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/draw_t_sne_50videos_20tasks_draw.py --version 3 \
  --highlight grab_roller:0.7:head place_burger_fries:0.8:head stack_blocks_three:0.7:head

#dp3
python /home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/draw_t_sne_50videos_20tasks_draw.py --version 3 \
  --highlight adjust_bottle:0.1:tail beat_block_hammer:0.2:tail click_bell:0.3:tail dump_bin_bigbin:0.2:tail handover_mic:0.3:tail move_can_pot:10:head move_playingcard_away:10:head open_laptop:0.2:tail pick_dual_bottles:0.9:head place_container_plate:0.1:tail put_bottles_dustbin:0.8:head put_object_cabinet:0.9:head shake_bottle:0.3:tail

#rdt
python /home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/draw_t_sne_50videos_20tasks_draw.py --version 3 \
  --highlight beat_block_hammer:0.8:head click_bell:0.7:head grab_roller:0.3:tail

#pi0
python /home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/draw_t_sne_50videos_20tasks_draw.py --version 3 \
  --highlight dump_bin_bigbin:0.8:head handover_mic:0.7:head open_laptop:0.8:head pick_dual_bottles:0.1:tail  place_container_plate:0.9:head put_object_cabinet:0.1:tail shake_bottle:0.7:head stack_blocks_three:0.3:tail

#cap
python /home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/draw_t_sne_50videos_20tasks_draw.py --version 3 \
  --highlight click_alarmclock:1.0:head lift_pot:0.3:tail place_burger_fries:0.2:tail place_empty_cup:0.4:tail put_bottles_dustbin:0.2:tail stack_blocks_two:1.0:head


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

