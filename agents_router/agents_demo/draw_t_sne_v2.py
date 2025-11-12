#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
t-SNE可视化脚本 - 支持3个版本的图
版本1：彩色散点 + 任务名称标注
版本2：彩色散点，无任务名称
版本3：灰色背景 + 可选红色高亮（通过参数指定）

读取 embeddings/*.json，做 L2 归一化 -> PCA -> t-SNE 到 2D，并画散点图。
"""

# ========= 参数区（按需修改） =========
EMBEDS_DIR   = "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache/embeddings"
OUTPUT_DIR   = "/data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache/figures"

USE_PCA      = True     # 先 PCA 再 t-SNE（更稳定）
PCA_DIM      = 50       # PCA 目标维度

# ===== t-SNE 核心参数 =====
TSNE_PERPLEXITY   = 60    # 困惑度：控制关注多少个邻居（推荐：5-100）
TSNE_N_ITER       = 2000  # 迭代次数：确保收敛（推荐：1000-3000）
TSNE_LEARNING     = 200.0 # 学习率：控制优化步长（推荐：10-1000）
TSNE_EARLY_EXAG   = 6.0   # 早期放大倍数：控制簇间距离（推荐：4-24）
RANDOM_STATE      = 42    # 固定随机种子（保证可复现）

# ===== 散点大小配置 =====
POINT_SIZE_V1 = 50   # 版本1的点大小（推荐：30-150）
POINT_SIZE_V2 = 50   # 版本2的点大小（推荐：30-150）
POINT_SIZE_V3_BG = 30  # 版本3背景灰点大小（推荐：20-80）
POINT_SIZE_V3_HL = 80  # 版本3高亮红点大小（推荐：50-200）

# ===== 图像尺寸配置 =====
FIGSIZE        = (12, 10)  # 图尺寸 (宽, 高) 英寸
DPI            = 180       # 分辨率（越高越清晰，但文件越大）

# 版本3：高亮任务列表（可以在命令行参数中覆盖）
HIGHLIGHT_TASKS = []  # 示例：["adjust_bottle", "open_laptop"]
# =====================================

import os, json, math, csv, argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return mat / n

def _load_all_embeddings(embed_dir: str):
    items = []
    for fn in sorted(os.listdir(embed_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(embed_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
            vec = j.get("vector")
            if not isinstance(vec, list) or len(vec) == 0:
                continue
            item = {
                "path": path,
                "vector": np.asarray(vec, dtype=float),
                "task": j.get("task", "unknown"),
                "model": j.get("model", "unknown"),
                "instruction": j.get("instruction", ""),
                "frame_path": j.get("frame_path", j.get("image_path", "")),
            }
            items.append(item)
        except Exception as e:
            print(f"[warn] 跳过 {path}: {e}")
    return items

def _auto_tsne_perplexity(n_samples: int, base: int) -> int:
    if n_samples <= 3:
        return 1
    return max(2, min(base, n_samples - 1, n_samples // 3 + 1))

def _get_colors(tasks, cmap_name='tab20'):
    """为每个任务分配颜色"""
    unique_tasks = sorted(set(tasks))
    
    # 如果任务数较少，使用tab10；否则使用tab20或hsv
    if len(unique_tasks) <= 10:
        cmap = plt.get_cmap('tab10')
    elif len(unique_tasks) <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        # 超过20个任务，使用连续色系
        cmap = plt.get_cmap('hsv')
    
    color_map = {}
    for i, task in enumerate(unique_tasks):
        if len(unique_tasks) <= 20:
            color_map[task] = cmap(i % cmap.N)
        else:
            color_map[task] = cmap(i / len(unique_tasks))
    
    return color_map, unique_tasks

def plot_version1(X2, tasks, output_prefix):
    """版本1：彩色散点 + 任务名称标注"""
    color_map, unique_tasks = _get_colors(tasks)
    colors = [color_map[t] for t in tasks]
    
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    # 绘制散点
    for task in unique_tasks:
        idx = np.array(tasks) == task
        ax.scatter(X2[idx, 0], X2[idx, 1], c=[color_map[task]], s=POINT_SIZE_V1,
                  label=task, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # 为每个任务添加文本标注（在簇的中心）
    for task in unique_tasks:
        idx = np.array(tasks) == task
        center_x = np.mean(X2[idx, 0])
        center_y = np.mean(X2[idx, 1])
        ax.text(center_x, center_y, task, fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    ax.set_xlabel("t-SNE dim 1", fontsize=12)
    ax.set_ylabel("t-SNE dim 2", fontsize=12)
    ax.set_title(f"t-SNE Visualization - Version 1 (N={len(tasks)} samples, {len(unique_tasks)} tasks)", 
                fontsize=14, fontweight='bold')
    
    # 图例放在右侧
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, 
             framealpha=0.9, title="Tasks")
    
    fig.tight_layout()
    fig_path = f"{output_prefix}_v1_with_labels.png"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] 版本1保存: {fig_path}")
    return fig_path

def plot_version2(X2, tasks, output_prefix):
    """版本2：彩色散点，无任务名称"""
    color_map, unique_tasks = _get_colors(tasks)
    colors = [color_map[t] for t in tasks]
    
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    # 绘制散点
    for task in unique_tasks:
        idx = np.array(tasks) == task
        ax.scatter(X2[idx, 0], X2[idx, 1], c=[color_map[task]], s=POINT_SIZE_V2,
                  label=task, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    ax.set_xlabel("t-SNE dim 1", fontsize=12)
    ax.set_ylabel("t-SNE dim 2", fontsize=12)
    ax.set_title(f"t-SNE Visualization - Version 2 (N={len(tasks)} samples, {len(unique_tasks)} tasks)", 
                fontsize=14, fontweight='bold')
    
    # 图例放在右侧
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, 
             framealpha=0.9, title="Tasks")
    
    fig.tight_layout()
    fig_path = f"{output_prefix}_v2_no_labels.png"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] 版本2保存: {fig_path}")
    return fig_path

def plot_version3(X2, tasks, highlight_tasks, output_prefix):
    """版本3：灰色背景 + 红色高亮指定任务"""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    
    tasks_array = np.array(tasks)
    
    # 所有点先画成灰色
    ax.scatter(X2[:, 0], X2[:, 1], c='lightgray', s=POINT_SIZE_V3_BG, alpha=0.5, 
              edgecolors='gray', linewidths=0.3, label='Other tasks')
    
    # 高亮指定的任务（红色）
    if highlight_tasks:
        for task in highlight_tasks:
            idx = tasks_array == task
            if np.any(idx):
                ax.scatter(X2[idx, 0], X2[idx, 1], c='red', s=POINT_SIZE_V3_HL, alpha=0.8,
                          edgecolors='darkred', linewidths=0.8, label=task)
    
    ax.set_xlabel("t-SNE dim 1", fontsize=12)
    ax.set_ylabel("t-SNE dim 2", fontsize=12)
    
    title = f"t-SNE Visualization - Version 3 (N={len(tasks)} samples)"
    if highlight_tasks:
        title += f"\nHighlighted: {', '.join(highlight_tasks)}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if highlight_tasks:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, 
                 framealpha=0.9, title="Highlighted Tasks")
    
    fig.tight_layout()
    fig_path = f"{output_prefix}_v3_highlight.png"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] 版本3保存: {fig_path}")
    return fig_path

def main():
    parser = argparse.ArgumentParser(description='t-SNE可视化 - 3个版本')
    parser.add_argument('--highlight', type=str, nargs='+', default=None,
                       help='版本3中要高亮显示的任务名称列表')
    parser.add_argument('--version', type=str, choices=['1', '2', '3', 'all'], default='all',
                       help='生成哪个版本的图（默认all生成全部）')
    args = parser.parse_args()
    
    _ensure_dir(OUTPUT_DIR)
    ts = datetime.now().strftime("%m%d%H%M%S")
    
    # 加载embeddings
    print("\n" + "="*70)
    print("加载 Embeddings")
    print("="*70)
    items = _load_all_embeddings(EMBEDS_DIR)
    if len(items) == 0:
        print(f"[错误] 未在 {EMBEDS_DIR} 找到 embedding json")
        return
    
    # 组装矩阵与标签
    X = np.vstack([it["vector"] for it in items])
    tasks = [it["task"] for it in items]
    models = [it["model"] for it in items]
    frames = [it["frame_path"] for it in items]
    files = [it["path"] for it in items]
    N, D = X.shape
    print(f"[信息] 载入 {N} 条向量，维度 {D}")
    
    unique_tasks = sorted(set(tasks))
    task_counts = {task: tasks.count(task) for task in unique_tasks}
    print(f"[信息] 任务数量: {len(unique_tasks)}")
    for task, count in task_counts.items():
        print(f"  - {task}: {count} 个样本")
    
    # L2归一化
    print(f"\n[信息] L2 归一化...")
    Xn = _l2_normalize(X)
    
    # PCA（可选）
    Xp = Xn
    pca_components = None
    if USE_PCA and min(N, D) > 2:
        n_comp = int(min(PCA_DIM, N, D))
        print(f"[信息] PCA 降维 -> {n_comp} 维")
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        Xp = pca.fit_transform(Xn)
        pca_components = n_comp
    
    # t-SNE
    perplexity = _auto_tsne_perplexity(N, TSNE_PERPLEXITY)
    print(f"[信息] t-SNE: perplexity={perplexity}, n_iter={TSNE_N_ITER}, random_state={RANDOM_STATE}")
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=TSNE_LEARNING,
        early_exaggeration=TSNE_EARLY_EXAG,
        init="pca",
        random_state=RANDOM_STATE,
        verbose=1,
    )
    X2 = tsne.fit_transform(Xp)
    print(f"[信息] ✓ t-SNE 完成")
    
    # 生成图像
    print("\n" + "="*70)
    print("生成可视化图像")
    print("="*70)
    
    output_prefix = os.path.join(OUTPUT_DIR, f"tsne_{ts}")
    highlight_tasks = args.highlight if args.highlight else HIGHLIGHT_TASKS
    
    generated_files = []
    
    if args.version == 'all' or args.version == '1':
        fig_path = plot_version1(X2, tasks, output_prefix)
        generated_files.append(fig_path)
    
    if args.version == 'all' or args.version == '2':
        fig_path = plot_version2(X2, tasks, output_prefix)
        generated_files.append(fig_path)
    
    if args.version == 'all' or args.version == '3':
        fig_path = plot_version3(X2, tasks, highlight_tasks, output_prefix)
        generated_files.append(fig_path)
    
    # 导出坐标 CSV
    csv_path = f"{output_prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "task", "model", "frame_path", "embedding_path"])
        for (x, y), t, m, fp, ep in zip(X2, tasks, models, frames, files):
            w.writerow([f"{x:.6f}", f"{y:.6f}", t, m, fp, ep])
    print(f"[✓] 坐标CSV保存: {csv_path}")
    generated_files.append(csv_path)
    
    # 保存 manifest
    manifest = {
        "timestamp": ts,
        "inputs_dir": EMBEDS_DIR,
        "output_files": generated_files,
        "N": N,
        "D": int(D),
        "num_tasks": len(unique_tasks),
        "task_counts": task_counts,
        "use_pca": USE_PCA,
        "pca_dim": pca_components if USE_PCA else None,
        "tsne": {
            "perplexity": perplexity,
            "n_iter": TSNE_N_ITER,
            "learning_rate": TSNE_LEARNING,
            "early_exaggeration": TSNE_EARLY_EXAG,
            "random_state": RANDOM_STATE,
            "metric": "euclidean+L2norm"
        },
        "highlight_tasks": highlight_tasks
    }
    jpath = f"{output_prefix}.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[✓] Manifest保存: {jpath}")
    
    print("\n" + "="*70)
    print("✓ 完成！")
    print("="*70)
    print(f"生成的文件:")
    for fpath in generated_files:
        print(f"  - {fpath}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

