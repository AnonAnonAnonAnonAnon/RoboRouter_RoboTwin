#!/bin/bash

# 运行单个ACT任务
# 用法: 
#   conda activate RoboTwin && bash run_act_single.sh <task_name> [gpu_id]

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到conda环境"
    echo "请先运行: conda activate RoboTwin"
    exit 1
fi

if [ "$CONDA_DEFAULT_ENV" != "RoboTwin" ]; then
    echo "警告: 当前conda环境是 $CONDA_DEFAULT_ENV，不是RoboTwin"
    echo "请运行: conda activate RoboTwin"
    exit 1
fi

if [ -z "$1" ]; then
    echo "用法: bash run_act_single.sh <task_name> [gpu_id]"
    echo ""
    echo "可用的任务:"
    echo "  - adjust_bottle"
    echo "  - beat_block_hammer"
    echo "  - click_alarmclock"
    echo "  - place_container_plate"
    echo "  - open_laptop"
    echo ""
    echo "示例:"
    echo "  bash run_act_single.sh adjust_bottle 0"
    exit 1
fi

TASK_NAME=$1
GPU_ID=${2:-0}
TASK_CONFIG="demo_clean"
EXPERT_DATA_NUM="50"

HF_DATASET="TianxingChen/RoboTwin2.0"

# 设置颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  RoboTwin ACT单任务测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}任务信息:${NC}"
echo -e "  任务名称: ${YELLOW}${TASK_NAME}${NC}"
echo -e "  任务配置: ${YELLOW}${TASK_CONFIG}${NC}"
echo -e "  GPU ID: ${YELLOW}${GPU_ID}${NC}"
echo ""

# 检查任务环境
if [ ! -f "./envs/${TASK_NAME}.py" ]; then
    echo -e "${RED}✗ 任务环境不存在: ./envs/${TASK_NAME}.py${NC}"
    exit 1
fi

# 检查依赖
python3 -c "import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}安装 huggingface_hub...${NC}"
    pip install huggingface_hub -q
fi

# 下载checkpoint
DOWNLOAD_DIR="./policy/ACT/checkpoints"
mkdir -p "${DOWNLOAD_DIR}"

CKPT_DIR="${DOWNLOAD_DIR}/act-${TASK_NAME}/demo_clean-50"
CKPT_FILE="${CKPT_DIR}/policy_last.ckpt"

if [ ! -f "${CKPT_FILE}" ]; then
    echo -e "${BLUE}下载ACT checkpoint...${NC}"
    echo -e "${YELLOW}提示: 如果下载慢，设置镜像: export HF_ENDPOINT=https://hf-mirror.com${NC}"
    python3 << EOF
from huggingface_hub import hf_hub_download
import os
import shutil

try:
    remote_path = f"act_ckpt/act-${TASK_NAME}/demo_clean-50/policy_last.ckpt"
    local_dir = "${DOWNLOAD_DIR}/act-${TASK_NAME}/demo_clean-50"
    
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"下载 ${TASK_NAME} ACT checkpoint (约336MB)...")
    
    # 下载到cache
    downloaded_file = hf_hub_download(
        repo_id="${HF_DATASET}",
        repo_type="dataset",
        filename=remote_path,
        resume_download=True
    )
    
    # 复制到目标位置
    target_file = os.path.join(local_dir, "policy_last.ckpt")
    shutil.copy2(downloaded_file, target_file)
    
    print(f"✓ 下载完成: {target_file}")
except Exception as e:
    print(f"✗ 下载失败: {e}")
    print("\n请尝试:")
    print("  export HF_ENDPOINT=https://hf-mirror.com")
    print("  然后重新运行")
    exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 下载失败${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Checkpoint已存在${NC}"
fi

# 验证checkpoint
if [ ! -f "${CKPT_FILE}" ]; then
    echo -e "${RED}✗ Checkpoint文件不存在: ${CKPT_FILE}${NC}"
    exit 1
fi

FILE_SIZE=$(du -h "${CKPT_FILE}" | cut -f1)
echo -e "${GREEN}✓ Checkpoint目录: ${CKPT_DIR}${NC}"
echo -e "${GREEN}✓ Checkpoint文件: ${CKPT_FILE} (${FILE_SIZE})${NC}"
echo ""

# 设置Xvfb渲染环境
export SAPIEN_OFFSCREEN_ONLY=1
export DISPLAY=:99

# 检查并启动Xvfb
if ! pgrep -x "Xvfb" > /dev/null; then
    echo -e "${YELLOW}启动Xvfb虚拟显示...${NC}"
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    sleep 2
    echo -e "${GREEN}✓ Xvfb已启动${NC}"
fi

# 运行评估
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo -e "${BLUE}开始评估...${NC}"
echo ""

PYTHONWARNINGS=ignore::UserWarning \
python3 << EOF
import sys
import yaml

sys.path.append("./")

# 读取ACT配置
with open("policy/ACT/deploy_policy.yml", "r") as f:
    config = yaml.safe_load(f)

# 更新配置
config.update({
    'task_name': '${TASK_NAME}',
    'task_config': '${TASK_CONFIG}',
    'ckpt_setting': 'act-${TASK_NAME}',
    'expert_data_num': '${EXPERT_DATA_NUM}',
    'seed': 0,
    'ckpt_dir': '${CKPT_DIR}',
    'policy_name': 'ACT'
})

# 运行评估
from script.eval_policy import main
main(config)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  ✓ 评估完成${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  ✗ 评估失败${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

