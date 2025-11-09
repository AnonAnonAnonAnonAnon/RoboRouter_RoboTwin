#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_tsne_embeddings.py
读取 embeddings/*.json（由 embed_batch_from_videos.py 产出），
做 L2 归一化 -> (可选) PCA -> t-SNE 到 2D，并画散点图。
同时导出坐标 CSV 和一个可复现实验的 manifest JSON。
"""

# ========= 参数区（按需修改） =========
EMBEDS_DIR   = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache/embeddings"
OUTPUT_DIR   = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache/figures"

USE_PCA      = True     # 先 PCA 再 t-SNE（更稳定）
PCA_DIM      = 50       # PCA 目标维度（会自动 min(实际维度,N样本数)）

TSNE_PERPLEXITY   = 15  # 会在代码里根据样本数自动调整不越界
TSNE_N_ITER       = 1000
TSNE_LEARNING     = "auto"
TSNE_EARLY_EXAG   = 12.0
RANDOM_STATE      = 42  # 固定随机种子便于复现

COLOR_BY    = "task"    # 颜色分组字段：task / model（如果有）
MARKER_BY   = None      # 形状分组字段：可设为 "model" 或 None

USE_THUMBNAILS = False  # 如要在点上贴首帧小图，设为 True（需要 PIL）
THUMB_SIZE     = 0.25   # 缩略图大小（以英寸为单位的近似）
FIGSIZE        = (7.2, 6.0)  # 图尺寸
DPI            = 180
# =====================================

from sklearn.manifold import TSNE

import os, json, math, csv
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return mat / n

def _load_all_embeddings(embed_dir: str):
    items = []
    for fn in sorted(os.listdir(embed_dir)):
        if not fn.endswith(".json"): continue
        path = os.path.join(embed_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
            vec = j.get("vector")
            if not isinstance(vec, list) or len(vec) == 0: continue
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
    # 取一个更稳妥的下界，且严格 < n_samples
    return max(2, min(base, n_samples - 1, n_samples // 3 + 1))

def _scatter_with_optional_thumbs(ax, X2, colors, labels_task, labels_marker, frames):
    if USE_THUMBNAILS:
        from PIL import Image
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        # 先画透明点用于图例，再贴图像
        ax.scatter(X2[:,0], X2[:,1], c=colors, s=1, alpha=0.0)
        for (x, y), fp in zip(X2, frames):
            if not fp or not os.path.isfile(fp): continue
            try:
                im = Image.open(fp)
                oi = OffsetImage(im, zoom=THUMB_SIZE/1.5)  # 调整体感大小
                ab = AnnotationBbox(oi, (x, y), frameon=False)
                ax.add_artist(ab)
            except Exception:
                pass
    else:
        markers = {"default":"o"}
        if labels_marker is not None:
            uniq = sorted(set(labels_marker))
            base_markers = ["o","s","^","D","P","v","*","X","h","<",">"]
            for i,u in enumerate(uniq):
                markers[u] = base_markers[i % len(base_markers)]
        else:
            uniq = ["default"]

        for u in uniq:
            idx = np.where(np.array(labels_marker if labels_marker else ["default"]*len(colors)) == u)[0]
            ax.scatter(X2[idx,0], X2[idx,1], c=np.array(colors)[idx], s=14,
                       marker=markers.get(u,"o"), label=(u if u!="default" else None), linewidths=0.3, edgecolors="none")

def main():
    _ensure_dir(OUTPUT_DIR)
    ts = datetime.now().strftime("%m%d%H%M%S")

    items = _load_all_embeddings(EMBEDS_DIR)
    if len(items) == 0:
        print(f"[err] 未在 {EMBEDS_DIR} 找到 embedding json")
        return

    # 组装矩阵与标签
    X = np.vstack([it["vector"] for it in items])        # (N,D)
    tasks = [it["task"] for it in items]
    models = [it["model"] for it in items]
    frames = [it["frame_path"] for it in items]
    files  = [it["path"] for it in items]
    N, D = X.shape
    print(f"[info] 载入 {N} 条向量，维度 {D}")

    # 归一化（欧氏 ~ 余弦）
    Xn = _l2_normalize(X)

    # (可选) PCA
    Xp = Xn
    pca_components = None
    if USE_PCA and min(N, D) > 2:
        n_comp = int(min(PCA_DIM, N, D))
        print(f"[info] PCA -> {n_comp} 维")
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        Xp = pca.fit_transform(Xn)
        pca_components = n_comp

    # t-SNE
    perplexity = _auto_tsne_perplexity(N, TSNE_PERPLEXITY)
    print(f"[info] t-SNE: perplexity={perplexity}, n_iter={TSNE_N_ITER}, random_state={RANDOM_STATE}")


    tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate=200.0,      # 数值更兼容；很多旧版不支持 "auto"
    early_exaggeration=TSNE_EARLY_EXAG,
    init="pca",
    random_state=RANDOM_STATE,
    verbose=1,
    # method="barnes_hut",    # 旧版默认就是它；如需可显式指定
    # angle=0.5,              # 需要时再加（旧版也支持）
    )
    X2 = tsne.fit_transform(Xp)    # 旧版 TSNE 迭代步数用默认（通常 1000）

    # 颜色映射
    group = tasks if COLOR_BY=="task" else (models if COLOR_BY=="model" else tasks)
    uniq = sorted(set(group))
    cmap = plt.get_cmap("tab10")
    color_map = {u: cmap(i % 10) for i,u in enumerate(uniq)}
    colors = [color_map[g] for g in group]

    # 形状映射
    marker_group = None
    if MARKER_BY == "model":
        marker_group = np.array(models)
    elif MARKER_BY == "task":
        marker_group = np.array(tasks)

    # 画图
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    _scatter_with_optional_thumbs(ax, X2, colors, np.array(tasks), marker_group, frames)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(f"t-SNE of multimodal embeddings (N={N})")
    # 图例：颜色按 COLOR_BY，形状按 MARKER_BY
    # 手动构造颜色图例
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0],[0], marker='o', color='w', label=str(u),
                            markerfacecolor=color_map[u], markersize=6) for u in uniq]
    legend1 = ax.legend(handles=color_handles, title=f"color: {COLOR_BY}", loc="best", fontsize=8)
    ax.add_artist(legend1)
    if MARKER_BY:
        uniq_m = sorted(set(marker_group))
        base_markers = ["o","s","^","D","P","v","*","X","h","<",">"]
        marker_handles = [Line2D([0],[0], marker=base_markers[i%len(base_markers)], color='k',
                                 label=str(u), linestyle='None', markersize=5) for i,u in enumerate(uniq_m)]
        ax.legend(handles=marker_handles, title=f"marker: {MARKER_BY}", loc="lower right", fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"tsne_{ts}.png")
    fig.savefig(fig_path)
    print(f"[ok] 保存图像: {fig_path}")

    # 导出坐标 CSV
    csv_path = os.path.join(OUTPUT_DIR, f"tsne_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x","y","task","model","frame_path","embedding_path"])
        for (x,y), t, m, fp, ep in zip(X2, tasks, models, frames, files):
            w.writerow([f"{x:.6f}", f"{y:.6f}", t, m, fp, ep])
    print(f"[ok] 保存坐标: {csv_path}")

    # 保存 manifest（记录参数，便于复现）
    manifest = {
        "timestamp": ts,
        "inputs_dir": EMBEDS_DIR,
        "figure": fig_path,
        "coords_csv": csv_path,
        "N": N,
        "D": int(D),
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
        "color_by": COLOR_BY,
        "marker_by": MARKER_BY
    }
    jpath = os.path.join(OUTPUT_DIR, f"tsne_{ts}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[ok] 保存参数: {jpath}")

if __name__ == "__main__":
    main()
