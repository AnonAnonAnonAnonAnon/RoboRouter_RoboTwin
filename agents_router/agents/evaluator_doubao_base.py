# doubao 实现vqa，输入视频的多帧以及元数据，输出执行情况总结
# 基础版本，只终端输出vqa_summary
# 实现读文件，并覆盖默认参数
#成功

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================== 导包区（按需修改） ==================
import os, sys, base64, mimetypes
from openai import OpenAI
import os, sys, base64, mimetypes, json

# ================== 配置区（按需修改） ==================
# 方舟接口与模型
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_ID = "doubao-seed-1-6-251015"   # 注意不要带换行或空格
API_KEY = "70f0d563-91a5-4704-a00e-f00cf3a9c864"  # 或留空用环境变量 ARK_API_KEY

# Evaluator 输入（按你当前约定）
FIRST_FRAME_PATH  = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/frames_to_push/f_0.jpg"
MIDDLE_FRAME_PATH = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/frames_to_push/f_80.jpg"
LAST_FRAME_PATH   = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/frames_to_push/f_399.jpg"  # 若无可换成 f_160.jpg

METADATA_PATH = "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/agents/cache/video_metadata_11081845.txt"

SUCCESS = False                 # 本次执行是否成功（由外部判定/上报）
DURATION_SEC = 40               # 执行时长（暂定固定 40s）
CHECK_ACTORS_CONTACT = False    # 规则函数：是否接触（暂定）
PLANAR_DELTA_OK = False          # 规则函数：|hammer_target_pose[:2] - block_pose[:2]| < [0.02, 0.02] ?

TASK_TEXT = "Grab hammer and beat the block."  # 本次任务指令

# ================== 代码区（一般无需改动） ==================
import os, sys, base64, mimetypes
from openai import OpenAI
import os, sys, base64, mimetypes, json


def get_api_key():
    k = (API_KEY or "").strip()
    if not k or k.startswith("<<<"):
        k = os.environ.get("ARK_API_KEY", "").strip()
    return k

def to_data_url(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到图片文件：{path}")
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif ext == ".png":
            mime = "image/png"
        elif ext == ".webp":
            mime = "image/webp"
        else:
            raise ValueError(f"无法识别的图片类型：{ext}；建议使用 jpg/png/webp")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_metadata(path: str) -> dict:
    """读取 txt；若不是完整 JSON，则自动补上花括号再解析。失败返回 {}。"""
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            s = f.read().strip()
        if not s:
            return {}
        # 允许 txt 里是裸键值对（没有最外层 {}）
        if not s.lstrip().startswith("{"):
            s = "{\n" + s + "\n}"
        return json.loads(s)
    except Exception as e:
        print(f"[warn] 读取/解析 metadata 失败：{e}", file=sys.stderr)
        return {}


def build_instruction() -> str:
    # 只产出一段英文 VQA 摘要；不做结构化；失败时明确失败表现。
    return (
        "你是评估器。根据给定的三帧图像（首帧/中帧/末帧）和外部信号，"
        "用英文写一段2-4句的简短摘要，说明本次任务执行过程与结果。"
        "若失败，请明确失败表现（如敲击偏移/未抓起/击中错误目标等）。"
        "只输出一个自然段，不要列表、标题、编号或多余格式；不要给建议或打分；不要臆测未出现的物体名称。"
        f"\n\n任务指令：{TASK_TEXT}"
        f"\n外部信号：success={SUCCESS}, duration={DURATION_SEC}s, "
        f"check_actors_contact={CHECK_ACTORS_CONTACT}, (hammer_target_pose[:2] - block_pose[:2]| < [0.02, 0.02])={PLANAR_DELTA_OK}。"
    )

def main():
        # 读取 metadata 并覆盖默认参数
    global SUCCESS, DURATION_SEC, CHECK_ACTORS_CONTACT, PLANAR_DELTA_OK, TASK_TEXT
    meta = load_metadata(METADATA_PATH)
    if "success" in meta:
        SUCCESS = bool(meta["success"])
    if "exec_time_s" in meta:
        try:
            DURATION_SEC = int(meta["exec_time_s"])
        except Exception:
            pass
    if "check_actors_contact" in meta:
        CHECK_ACTORS_CONTACT = bool(meta["check_actors_contact"])
    # 允许可选字段：planar_delta_ok / task_text
    if "planar_delta_ok" in meta:
        PLANAR_DELTA_OK = bool(meta["planar_delta_ok"])
    if "task_text" in meta and str(meta["task_text"]).strip():
        TASK_TEXT = str(meta["task_text"]).strip()

    key = get_api_key()
    if not key:
        print("未找到 API Key：请在脚本顶部填写 API_KEY，或先 `export ARK_API_KEY=...`")
        sys.exit(1)

    # 三帧按“首→中→末”顺序送入
    frame_urls = [
        ("首帧",   to_data_url(FIRST_FRAME_PATH)),
        ("中间帧", to_data_url(MIDDLE_FRAME_PATH)),
        ("末帧",   to_data_url(LAST_FRAME_PATH)),
    ]

    # 组装消息：在每张图前用一小段文字标注身份，便于模型理解时间顺序
    contents = []
    for label, url in frame_urls:
        contents.append({"type": "text", "text": f"{label}："})
        contents.append({"type": "image_url", "image_url": {"url": url}})
    contents.append({"type": "text", "text": build_instruction()})

    client = OpenAI(base_url=ARK_BASE_URL, api_key=key)
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": contents}],
        reasoning_effort="medium",
    )

    msg = resp.choices[0].message
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        print("----- reasoning -----")
        print(msg.reasoning_content)
    print("----- vqa_summary -----")
    print(msg.content)

    if getattr(resp, "usage", None):
        print("----- usage -----")
        print(resp.usage)

if __name__ == "__main__":
    main()
