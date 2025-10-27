#未能成功实现
# 

# -*- coding: utf-8 -*-
"""
Minimal video anomaly detector with OpenAI Agents SDK

"""

import os
from pathlib import Path

from urllib.parse import urlparse, quote

from urllib.parse import urlparse, quote

import requests

import tempfile, os

import cv2
import sys
import json
import base64
import argparse
import asyncio
from typing import List, Tuple, Literal
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from agents import (
    Agent, Runner,
    set_default_openai_client, set_default_openai_api, set_tracing_disabled,
)

# ===== ① 你的网关与密钥（保持你能跑通的设置）=====
BASE_URL = "https://api.chatanywhere.tech/v1"
API_KEY  = "sk-AhGuNmK6xnFGdBCkFGpG0lcqj3TgLT7dQKU5JUSpaNQkUpZV"
MODEL    = "gpt-4o-mini"         # 用 /v1/models 里存在的模型
API_KIND = "chat_completions"    # 走 chat_completions，用 image_url# ================================================

set_tracing_disabled(True)
set_default_openai_api(API_KIND)
set_default_openai_client(AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY))


# ---------------------------------------
# 1) 结构化输出 Schema
# ---------------------------------------
class AnomalyResult(BaseModel):
    verdict: Literal["success", "fail", "uncertain"] = Field(..., description="Overall outcome")
    summary: str = Field(..., description="Brief failure manifestation; if success, say 'no anomaly observed'")
    key_frames: List[int] = Field(..., description="Evidence frame indices from provided list")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0..1")


# ---------------------------------------
# 2) 视频采样：首/尾 + N 个中间帧
# ---------------------------------------
def sample_frames(video_path: str, num_mid_frames: int = 4) -> List[Tuple[int, bytes]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Empty or unreadable video (frame_count=0).")

    indices = [0]
    if num_mid_frames > 0 and total > 2:
        step = (total - 1) / (num_mid_frames + 1)
        mids = [int(round(step * (i + 1))) for i in range(num_mid_frames)]
        mids = [min(max(1, m), total - 2) for m in mids]
        indices.extend(mids)
    if total > 1:
        indices.append(total - 1)

    indices = sorted(set(indices))
    frames: List[Tuple[int, bytes]] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        frames.append((idx, buf.tobytes()))

    cap.release()
    if not frames:
        raise RuntimeError("Failed to sample any frame from the video.")
    return frames


# ---------------------------------------
# 3) 构造多模态输入项（文本 + 多图）
# ---------------------------------------
import time


def save_frames_to_dir(frames: List[Tuple[int, bytes]], outdir: str) -> List[Tuple[int, str]]:
    """
    把采样帧保存到 outdir，文件名固定 f_{idx}.jpg
    返回 [(idx, local_path), ...]
    """
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    results: List[Tuple[int, str]] = []
    for idx, jpg in frames:
        p = d / f"f_{idx}.jpg"
        with open(p, "wb") as f:
            f.write(jpg)
        print(f"[export] saved -> {p}")
        results.append((idx, str(p)))
    return results

def make_urls_from_base(task: str, indices: List[int], base_url: str) -> list:
    """
    用 base_url 拼出直链：{base_url}/f_{idx}.jpg
    返回给 make_input_items_from_urls 的 [(idx, url), ...]
    """
    base = base_url.rstrip("/")
    urls = [(idx, f"{base}/f_{idx}.jpg") for idx in indices]
    return urls

def upload_frames_get_urls(frames: List[Tuple[int, bytes]]) -> List[Tuple[int, str]]:
    """
    仅使用 catbox 永久上传，返回 [(frame_index, https://files.catbox.moe/xxxx.jpg), ...]
    加 3 次重试，避免偶发断连。
    """
    ua = {"User-Agent": "curl/8.5.0", "Accept": "*/*"}
    urls: List[Tuple[int, str]] = []

    for idx, jpg in frames:
        last_err = None
        for attempt in range(1, 4):
            try:
                r = requests.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": (f"f_{idx}.jpg", jpg, "image/jpeg")},
                    headers=ua, timeout=30
                )
                r.raise_for_status()
                url = r.text.strip()
                if url.startswith("https://files.catbox.moe/"):
                    print(f"[debug] frame {idx} -> {url}")
                    urls.append((idx, url))
                    break
                else:
                    raise RuntimeError(f"catbox unexpected resp: {r.text[:200]}")
            except Exception as e:
                last_err = e
                print(f"[warn] catbox upload idx={idx} attempt={attempt} -> {e}")
                time.sleep(1.5)
        else:
            # 3 次都失败
            raise RuntimeError(f"catbox upload failed for frame {idx}: {last_err}")
    return urls



def _proxy_image(raw_url: str) -> str:
    """
    把 https://host/path 转成 https://wsrv.nl/?url=ssl:host/path
    避免 ChatAnyWhere 无法直接访问原图域名的问题。
    """
    p = urlparse(raw_url)
    path = p.path + (f"?{p.query}" if p.query else "")
    proxied = f"ssl:{p.netloc}{path}"
    return "https://wsrv.nl/?url=" + quote(proxied, safe=":/?&=._-")

def _probe_url(url: str, label: str = ""):
    try:
        r = requests.get(url, timeout=15, stream=True)
        ct = r.headers.get("Content-Type", "?")
        cl = r.headers.get("Content-Length", "?")
        peek = b"".join(r.iter_content(512, decode_unicode=False))[:64]
        print(f"[probe] {label} {url}\n        status={r.status_code} ct={ct} len={cl} head={peek[:16]!r}")
    except Exception as e:
        print(f"[probe-err] {label} {url} -> {e}")

def make_input_items_from_urls(task: str, urls: List[Tuple[int, str]], use_proxy: bool = True) -> list:
    text = (
        "You are a robot-operation anomaly inspector.\n"
        f"Task: {task}\n"
        f"You will receive {len(urls)} frames sampled from a single execution video "
        "(first, middle, last). Determine if the operation succeeded, failed, or is uncertain.\n"
        "If failed, summarize the failure manifestation in 1-2 concise sentences.\n"
        "Return ONLY JSON with fields: verdict (success|fail|uncertain), summary (string), "
        "key_frames (array of indices from the provided list), confidence (0..1)."
    )

    content_items = [{"type": "input_text", "text": text}]
    for (idx, raw_url) in urls:
        url_for_model = _proxy_image(raw_url) if use_proxy else raw_url
        # 打印本地探测：原图 & 最终送入的图
        _probe_url(raw_url, label=f"raw[{idx}]")
        _probe_url(url_for_model, label=f"send[{idx}]")

        content_items.append({"type": "input_image", "image_url": {"url": url_for_model}})
        content_items.append({"type": "input_text", "text": f"frame_index={idx}"})
    return [{"role": "user", "content": content_items}]

# ---------------------------------------
# 4) Analyzer Agent（单 Agent）
# ---------------------------------------
analyzer = Agent(
    name="Video Anomaly Analyzer",
    instructions=(
        "You receive key frames from a single robot operation.\n"
        "- Decide verdict: success / fail / uncertain.\n"
        "- If fail, briefly summarize failure manifestation (e.g., grasp offset, slip, collision, "
        "misalignment, missing part, jamming). If success, say 'no anomaly observed'.\n"
        "- Choose key_frames as the indices (integers) of the evidence frames provided by the user.\n"
        "- Provide confidence in [0,1].\n"
        "Respond with JSON only, strictly matching the given schema."
    ),
    model=MODEL,
    output_type=AnomalyResult,
)


# ---------------------------------------
# 5) CLI / main
# ---------------------------------------
async def run_once(video_path: str, num_mid: int, task: str,
                   use_proxy: bool, debug_url: str | None,
                   export_dir: str | None, base_url: str | None,
                   skip_analyze: bool):

    if debug_url:
        urls = [(0, debug_url)]
        input_items = make_input_items_from_urls(task, urls, use_proxy=use_proxy)
        print(f"[info] debug-url mode: use_proxy={use_proxy}")
        result = await Runner.run(analyzer, input_items, max_turns=1)
        output: AnomalyResult = result.final_output
        print(json.dumps(output.model_dump(), ensure_ascii=False, indent=2))
        return

    # 正常路径：采样
    frames = sample_frames(video_path, num_mid_frames=num_mid)
    sampled_indices = [idx for idx, _ in frames]
    print(f"[info] sampled frame indices: {sampled_indices}")

    # 如果给了 base_url，则本地导出 -> 你自己 push -> 用 base_url 直链
    if base_url:
        if not export_dir:
            export_dir = "./frames_to_push"
        save_frames_to_dir(frames, export_dir)
        urls = make_urls_from_base(task, sampled_indices, base_url)
        for i, u in urls:
            print(f"[url] frame {i} -> {u}")
        if skip_analyze:
            print("[info] skip_analyze=True, 仅导出并打印直链。请把上述文件 push 到对应路径后再运行分析。")
            return
        # 继续做探针 + 分析（注意：若你还没 push，此处会 404）
        input_items = make_input_items_from_urls(task, urls, use_proxy=use_proxy)
        result = await Runner.run(analyzer, input_items, max_turns=1)
        output: AnomalyResult = result.final_output
        print(json.dumps(output.model_dump(), ensure_ascii=False, indent=2))
        return

    # 否则，走原有上传（不推荐，因为你这边经常失败）
    urls = upload_frames_get_urls(frames)
    for i, u in urls:
        print(f"[debug] frame {i} -> {u}")
    input_items = make_input_items_from_urls(task, urls, use_proxy=use_proxy)
    result = await Runner.run(analyzer, input_items, max_turns=1)
    output: AnomalyResult = result.final_output
    print(json.dumps(output.model_dump(), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Minimal video anomaly detector (Agents SDK)")
    
    parser.add_argument(
        "video",
        nargs="?",
        default="/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents/video/fail_exec_video/beat_wrong_place.mp4",
        type=str,
        help="Path to input video file (optional; uses default if omitted)",
    )
    
    parser.add_argument("--num-mid", type=int, default=4, help="Number of middle frames (default: 4)")
    parser.add_argument("--task", type=str, default="beat_block_hammer", help="Task description")

    parser.add_argument("--debug-url", type=str, default=None, help="Skip sampling/upload; send this URL as image input")

    parser.add_argument("--no-proxy", dest="no_proxy", action="store_true", help="Do not proxy image URLs via wsrv")

    parser.add_argument("--export-dir", type=str, default=None,
                    help="导出采样帧到该目录（文件名 f_{idx}.jpg）")

    parser.add_argument("--base-url", type=str, default=None,
                        help="直链基础 URL，例如：https://raw.githubusercontent.com/<owner>/<repo>/<branch>/path/to/frames")

    parser.add_argument("--skip-analyze", action="store_true",
                        help="仅导出并打印直链，不发起分析（用于先 push 再跑分析）")


    args = parser.parse_args()

    try:
        asyncio.run(run_once(
            args.video, args.num_mid, args.task,
            use_proxy=(not args.no_proxy), debug_url=args.debug_url,
            export_dir=args.export_dir, base_url=args.base_url,
            skip_analyze=args.skip_analyze
        ))
    except KeyboardInterrupt:
        print("\n[warn] interrupted by user.", file=sys.stderr)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
