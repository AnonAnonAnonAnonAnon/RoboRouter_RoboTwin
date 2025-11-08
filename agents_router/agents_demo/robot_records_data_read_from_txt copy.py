# -*- coding: utf-8 -*-
"""
机器人任务记录数据（从 cache/*.txt 动态生成）
- 每个 txt 作为一条记录：description=文件全文
- 从文件名解析 model/任务；可从文本中抓取“xx%”作为 success_rate（可缺省）
"""

from typing import List, Dict
import os, re

# 你的缓存目录（放 txt 的地方）
CACHE_DIR = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents/cache"

# 显式列出要读的文件（也可用目录自动匹配，见下）
TXT_FILES = [
    os.path.join(CACHE_DIR, "act_block_hammer_11081845.txt"),
    os.path.join(CACHE_DIR, "dp_block_hammer_11081845.txt"),
    os.path.join(CACHE_DIR, "dp3_block_hammer_11081845.txt"),
    os.path.join(CACHE_DIR, "rdt_block_hammer_11081845 copy.txt"),
]

# 文件名前缀 -> ckpt 名（可按需改）
MODEL2CKPT = {
    "act": "act_ckpt_auto",
    "dp": "dp_ckpt_auto",
    "dp3": "dp3_ckpt_auto",
    "rdt": "rdt_ckpt_auto",
}

# 文件名中的任务片段 -> 规范任务名
TASK_MAP = {
    "block_hammer": "beat_block_hammer",
    "open_laptop":  "open_laptop",
    "place_container_plate": "place_container_plate",
    # 有新的任务可在这里补
}

def _parse_model_and_task_from_filename(path: str):
    name = os.path.basename(path).lower().replace(" ", "_")
    # 允许形如: act_block_hammer_11081845.txt
    m = re.match(r'^(act|dp3|dp|rdt)_([a-z0-9_]+?)_[0-9]+', name)
    model = m.group(1) if m else "unknown"
    task_key = m.group(2) if m else "unknown"
    task = TASK_MAP.get(task_key, task_key)
    return model, task

def _parse_success_rate_from_text(text: str):
    # 从文本里抓第一个“NN%”当成功率（没有就返回 None）
    m = re.search(r'(\d{1,3})\s*%', text)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return round(v / 100.0, 4)
    return None

def _load_one_txt(path: str, rid: int) -> Dict:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
    model, task = _parse_model_and_task_from_filename(path)
    ckpt = MODEL2CKPT.get(model, f"{model}_ckpt")
    sr = _parse_success_rate_from_text(content)
    return {
        "id": rid,
        "task": task,
        "ckpt": ckpt,
        "success_rate": sr if sr is not None else 0.0,  # 可缺省，用 0.0 兜底
        "notes": f"Imported from {os.path.basename(path)}",
        "description": content,
    }

def _gather_txt_files() -> List[str]:
    # 如果你想自动读取目录下所有形如 model_task_timestamp.txt 的文件，打开这段注释并用它替换 TXT_FILES
    # patt = re.compile(r'^(act|dp3|dp|rdt)_[a-z0-9_]+_[0-9]+\.txt$', re.I)
    # return [os.path.join(CACHE_DIR, fn) for fn in os.listdir(CACHE_DIR) if patt.match(fn)]
    return [p for p in TXT_FILES if os.path.isfile(p)]

def _load_records_from_txts(paths: List[str]) -> List[Dict]:
    records: List[Dict] = []
    rid = 1
    for p in paths:
        try:
            records.append(_load_one_txt(p, rid))
            rid += 1
        except Exception as e:
            # 柔性容错：坏文件跳过
            print(f"[warn] 跳过无法读取的文件 {p}: {e}")
    return records

# === 对外暴露：RECORDS ===
RECORDS: List[Dict] = _load_records_from_txts(_gather_txt_files())
