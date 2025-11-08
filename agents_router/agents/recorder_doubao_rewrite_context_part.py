#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recorder: read metadata -> ask Doubao to choose task -> update SR x/y in router_context_SR_part.json
"""

# ===== 配置区（可用命令行参数覆盖） =====
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_ID = "doubao-seed-1-6-251015"
API_KEY = "70f0d563-91a5-4704-a00e-f00cf3a9c864"  # 建议用环境变量 ARK_API_KEY

METADATA_PATH = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents/cache/video_metadata_11081845.txt"
CONTEXT_JSON_PATH = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents/cache/router_context_SR_part.json"

# 本次执行对应的“策略/模型名”，会更新该模型在目标任务下的 SR。
EXEC_MODEL = "DP3"   # 例如：RDT / Pi0 / ACT / DP / DP3

# ===== 代码区 =====
import os, sys, json, shutil, argparse, re
from datetime import datetime
from openai import OpenAI

def get_api_key():
    k = (API_KEY or "").strip()
    if not k or k.startswith("<<<"):
        k = os.environ.get("ARK_API_KEY", "").strip()
    return k

def read_text_metadata(path: str) -> dict:
    """读取 txt（可能是裸键值对），尽量解析成 JSON dict。"""
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            s = f.read().strip()
        if not s:
            return {}
        if not s.lstrip().startswith("{"):
            s = "{\n" + s + "\n}"
        return json.loads(s)
    except Exception as e:
        print(f"[warn] 读取/解析 metadata 失败：{e}", file=sys.stderr)
        return {}

def load_context(path: str) -> dict:
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # 初始化一个空骨架
        return {"setting": "Easy", "tasks": {}}
    except Exception as e:
        print(f"[err] 读取 context JSON 失败：{e}", file=sys.stderr)
        sys.exit(2)

def save_context_with_backup(path: str, data: dict):
    dpath = os.path.dirname(os.path.expanduser(path))
    os.makedirs(dpath, exist_ok=True)
    # 备份
    if os.path.exists(path):
        ts = datetime.now().strftime("%m%d%H%M%S")
        bak = f"{path}.bak_{ts}"
        shutil.copy2(path, bak)
        print(f"[info] 已备份原文件 -> {bak}")
    # 写回
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[ok] 已写回 -> {path}")

def parse_fraction(v) -> tuple[int, int]:
    """将 '81/100' 或数字/None 转为 (succ, att)；空时返回 (0,0)。"""
    if v is None:
        return 0, 0
    if isinstance(v, (int, float)):
        # 若有人直接存了成功次数，视作 (v, v) 不合理，这里规范化为 (v, ?)
        return int(v), 0
    if isinstance(v, str):
        m = re.match(r'^\s*(\d+)\s*/\s*(\d+)\s*$', v)
        if m:
            return int(m.group(1)), int(m.group(2))
    return 0, 0

def to_fraction_str(succ: int, att: int) -> str:
    att = max(att, 0)
    succ = max(min(succ, att), 0) if att > 0 else max(succ, 0)
    return f"{succ}/{att}"

def call_doubao_choose_task_and_success(meta: dict, task_list: list[str], default_success: bool) -> tuple[str, bool]:
    """
    让豆包在 task_list 中精确选择一个任务名，并确认 success。
    返回 (task_name, success_bool)；若失败则回退到启发式。
    """
    key = get_api_key()
    client = OpenAI(base_url=ARK_BASE_URL, api_key=key)

    # 构造提示：提供可选任务清单，要求“只从清单中选”，输出严格 JSON
    meta_compact = {k: meta[k] for k in meta if k in ("obj", "success", "exec_time_s", "check_actors_contact", "task_text")}
    user_prompt = (
        "You are a recorder. Given the metadata of a robotic manipulation episode and a closed set of task names, "
        "choose exactly ONE task name from the provided list that best matches this episode, and confirm success as a boolean.\n\n"
        f"Allowed task names (exact match, choose one): {task_list}\n"
        f"Metadata (JSON): {json.dumps(meta_compact, ensure_ascii=False)}\n\n"
        "Return MUST be a valid JSON object with keys: "
        "{\"task\": <string from list>, \"success\": <true|false>} and nothing else."
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": user_prompt}],
            reasoning_effort="medium",
        )
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        # 尝试只提取 JSON
        try:
            start = text.find("{")
            end = text.rfind("}")
            j = json.loads(text[start:end+1])
            task = str(j.get("task", "")).strip()
            succ = bool(j.get("success", default_success))
            if task in task_list:
                return task, succ
            else:
                print(f"[warn] 模型返回的 task 不在列表中：{task}，将回退启发式。")
        except Exception as e:
            print(f"[warn] 解析模型 JSON 失败：{e}；原文：{text[:200]}...")
    except Exception as e:
        print(f"[warn] 调用豆包失败：{e}")

    # 回退启发式（尽量不用）
    objs = [o.lower() for o in meta.get("obj", []) if isinstance(o, str)]
    guess = None
    if {"hammer", "block"}.issubset(set(objs)):
        # 优先匹配 Beat Block Hammer（若存在）
        for t in task_list:
            if "beat" in t.lower() and "hammer" in t.lower() and "block" in t.lower():
                guess = t
                break
    if guess is None and task_list:
        guess = task_list[0]
    return guess, bool(meta.get("success", default_success))

def update_sr(context: dict, task: str, model: str, success: bool) -> dict:
    context.setdefault("tasks", {})
    context["tasks"].setdefault(task, {})
    cur = context["tasks"][task].get(model, "0/0")
    s, a = parse_fraction(cur)
    a += 1
    if success:
        s += 1
    context["tasks"][task][model] = to_fraction_str(s, a)
    return context

def main():
    parser = argparse.ArgumentParser(description="Record episode result into router_context_SR_part.json")
    parser.add_argument("--metadata", default=METADATA_PATH, help="path to video metadata txt")
    parser.add_argument("--context", default=CONTEXT_JSON_PATH, help="path to SR context json")
    parser.add_argument("--model", default=EXEC_MODEL, help="model/policy name to update, e.g., DP3")
    args = parser.parse_args()

    # 读取
    meta = read_text_metadata(args.metadata)
    if not meta:
        print("[err] metadata 为空/解析失败")
        sys.exit(2)

    context = load_context(args.context)
    task_list = sorted(list((context.get("tasks") or {}).keys()))
    if not task_list:
        print("[warn] context 中没有任务清单，将创建一个最小任务清单用于冷启动。")
        # 你可以改成自己项目的常用任务
        task_list = [
            "Beat Block Hammer", "Adjust Bottle", "Click Alarmclock",
            "Open Laptop", "Place Container Plate"
        ]
        context.setdefault("tasks", {})
        for t in task_list:
            context["tasks"].setdefault(t, {})

    # 调用豆包选择任务 & 成功与否
    task, success = call_doubao_choose_task_and_success(meta, task_list, default_success=bool(meta.get("success", False)))
    print(f"[info] 选择的任务: {task} | success={success} | model={args.model}")

    # 更新 SR
    context = update_sr(context, task, args.model, success)

    # 写回（带备份）
    save_context_with_backup(args.context, context)

    # 友好打印当前任务的各模型 SR
    row = context["tasks"][task]
    nice = ", ".join(f"{k} {v}" for k, v in row.items())
    print(f"[SR] {task}: {nice}")

if __name__ == "__main__":
    # 环境检查
    if not get_api_key():
        print("未找到 API Key：请在脚本顶部填写 API_KEY，或先 `export ARK_API_KEY=...`")
        sys.exit(1)
    main()
