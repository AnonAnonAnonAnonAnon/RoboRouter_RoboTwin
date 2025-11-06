# -*- coding: utf-8 -*-
"""
机器人任务记录数据
- 存储各种任务的checkpoint配置和成功率数据
"""

from typing import List, Dict

# 机器人任务记录数据库
RECORDS: List[Dict] = [
    {
        "id": 1,
        "task": "open_laptop",
        "ckpt": "act_ckpt_v1",
        "success_rate": 0.82,
        "notes": "抓取角度敏感，需要精确定位铰链位置",
        "description": "使用ACT策略打开笔记本电脑，成功率82%。关键点：抓取角度敏感，需要精确定位铰链位置。适合光照稳定的环境。"
    },
    {
        "id": 2,
        "task": "open_laptop",
        "ckpt": "dp_ckpt_v3",
        "success_rate": 0.76,
        "notes": "对光照更稳定，鲁棒性好",
        "description": "使用扩散策略打开笔记本电脑，成功率76%。特点：对光照变化更稳定，鲁棒性好，适合复杂环境。"
    },
    {
        "id": 3,
        "task": "beat_block_hammer",
        "ckpt": "dp_ckpt_v5",
        "success_rate": 0.71,
        "notes": "需更高频视觉反馈",
        "description": "使用扩散策略v5抓取锤子敲击木块，成功率71%。需要更高频率的视觉反馈来调整轨迹，对力控要求较高。"
    },
    {
        "id": 4,
        "task": "beat_block_hammer",
        "ckpt": "act_ckpt_v4",
        "success_rate": 0.68,
        "notes": "轨迹平滑但精度略低",
        "description": "使用ACT策略v4完成锤子敲击任务，成功率68%。轨迹平滑，但终点精度略低，适合对位置要求不严格的场景。"
    },
    {
        "id": 5,
        "task": "place_container_plate",
        "ckpt": "act_ckpt_v2",
        "success_rate": 0.65,
        "notes": "控制器需微调，对容器形状敏感",
        "description": "将容器放置到盘子上，使用ACT策略v2，成功率65%。控制器需要针对不同容器形状进行微调。"
    },
    {
        "id": 6,
        "task": "grasp_small_object",
        "ckpt": "dp_ckpt_v6",
        "success_rate": 0.88,
        "notes": "精细抓取，适合小物体",
        "description": "抓取小型物体（如螺丝、硬币），使用扩散策略v6，成功率88%。专门优化了精细抓取能力，夹持力控制精准。"
    },
]

