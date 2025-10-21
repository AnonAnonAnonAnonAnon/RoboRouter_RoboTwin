#!/bin/bash

# 批量下载并测试多个RoboTwin模型
# 用法: 
#   方式1 (推荐): conda activate RoboTwin && bash run_all_models.sh [GPU_ID]
#   方式2: conda run -n RoboTwin bash run_all_models.sh [GPU_ID]

GPU_ID=${1:-0}
HF_REPO="Avada11/RoboTwin-Model-AgileX-DP"

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "警告: 未检测到conda环境"
    echo "请先运行: conda activate RoboTwin"
    echo "然后再运行此脚本"
    exit 1
fi

if [ "$CONDA_DEFAULT_ENV" != "RoboTwin" ]; then
    echo "警告: 当前conda环境是 $CONDA_DEFAULT_ENV，不是RoboTwin"
    echo "请运行: conda activate RoboTwin"
    exit 1
fi

# 定义所有要测试的任务
declare -a TASKS=(
    "adjust_bottle"
    "beat_block_hammer"
    "click_alarmclock"
    "place_container_plate"
    "open_laptop"
)

# 配置参数
TASK_CONFIG="demo_clean"
EXPERT_DATA_NUM="100"
CHECKPOINT_SUBDIR_SUFFIX="-demo_clean-100/100"
CHECKPOINT_FILE="500.ckpt"
SEED="0"

# 设置颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${MAGENTA}============================================${NC}"
echo -e "${MAGENTA}  RoboTwin 批量模型测试${NC}"
echo -e "${MAGENTA}============================================${NC}"
echo ""
echo -e "${CYAN}HuggingFace仓库:${NC} ${YELLOW}${HF_REPO}${NC}"
echo -e "${CYAN}任务配置:${NC} ${YELLOW}${TASK_CONFIG}${NC}"
echo -e "${CYAN}GPU ID:${NC} ${YELLOW}${GPU_ID}${NC}"
echo -e "${CYAN}待测试任务数量:${NC} ${YELLOW}${#TASKS[@]}${NC}"
echo ""

# 检查Python依赖
echo -e "${BLUE}[准备]${NC} 检查依赖..."
python3 -c "import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}  安装 huggingface_hub...${NC}"
    pip install huggingface_hub -q
fi
echo -e "${GREEN}  ✓ 依赖检查完成${NC}"
echo ""

# 下载所有模型
DOWNLOAD_DIR="./policy/DP/checkpoints/downloaded_${HF_REPO//\//_}"

echo -e "${BLUE}[下载]${NC} 从HuggingFace下载所有模型..."
echo -e "${YELLOW}  目标目录: ${DOWNLOAD_DIR}${NC}"
echo ""

# 检查是否已经下载过
NEED_DOWNLOAD=false
for task in "${TASKS[@]}"; do
    CKPT_PATH="${DOWNLOAD_DIR}/${task}${CHECKPOINT_SUBDIR_SUFFIX}/${CHECKPOINT_FILE}"
    if [ ! -f "${CKPT_PATH}" ]; then
        NEED_DOWNLOAD=true
        break
    fi
done

if [ "$NEED_DOWNLOAD" = true ]; then
    echo -e "${YELLOW}  正在下载5个checkpoint（约7.5GB，仅需要的文件）...${NC}"
    python3 << EOF
from huggingface_hub import snapshot_download
import os

# 只下载5个需要的checkpoint文件
checkpoint_patterns = [
    "adjust_bottle-demo_clean-100/100/500.ckpt",
    "beat_block_hammer-demo_clean-100/100/500.ckpt",
    "click_alarmclock-demo_clean-100/100/500.ckpt",
    "place_container_plate-demo_clean-100/100/500.ckpt",
    "open_laptop-demo_clean-100/100/500.ckpt"
]

try:
    print("  开始下载必需的checkpoint文件...")
    
    for i, pattern in enumerate(checkpoint_patterns, 1):
        task_name = pattern.split('-')[0].replace('_', ' ').title()
        print(f"  [{i}/5] {task_name}...")
        
        snapshot_download(
            repo_id="${HF_REPO}",
            allow_patterns=[pattern],
            local_dir="${DOWNLOAD_DIR}",
            local_dir_use_symlinks=False,
            resume_download=True
        )
    
    print("  ✓ 所有checkpoint下载完成")
except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 下载失败${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}  ✓ 所有模型已存在，跳过下载${NC}"
fi

echo ""

# 验证所有checkpoint文件
echo -e "${BLUE}[验证]${NC} 检查checkpoint文件..."
MISSING_FILES=0
for task in "${TASKS[@]}"; do
    CKPT_PATH="${DOWNLOAD_DIR}/${task}${CHECKPOINT_SUBDIR_SUFFIX}/${CHECKPOINT_FILE}"
    if [ -f "${CKPT_PATH}" ]; then
        FILE_SIZE=$(du -h "${CKPT_PATH}" | cut -f1)
        echo -e "${GREEN}  ✓${NC} ${task}: ${FILE_SIZE}"
    else
        echo -e "${RED}  ✗${NC} ${task}: 文件不存在"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${RED}✗ 有 ${MISSING_FILES} 个checkpoint文件缺失${NC}"
    exit 1
fi

echo ""

# 创建结果摘要文件
RESULTS_DIR="./eval_result/batch_results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
SUMMARY_FILE="${RESULTS_DIR}/summary_${TIMESTAMP}.txt"

echo "RoboTwin 批量测试结果" > "${SUMMARY_FILE}"
echo "时间: $(date)" >> "${SUMMARY_FILE}"
echo "HuggingFace仓库: ${HF_REPO}" >> "${SUMMARY_FILE}"
echo "任务配置: ${TASK_CONFIG}" >> "${SUMMARY_FILE}"
echo "GPU ID: ${GPU_ID}" >> "${SUMMARY_FILE}"
echo "======================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# 设置Xvfb渲染环境
export SAPIEN_OFFSCREEN_ONLY=1
export DISPLAY=:99

# 检查并启动Xvfb
if ! pgrep -x "Xvfb" > /dev/null; then
    echo -e "${YELLOW}启动Xvfb虚拟显示...${NC}"
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    sleep 2
    echo -e "${GREEN}  ✓ Xvfb已启动${NC}"
    echo ""
fi

# 测试所有模型
export CUDA_VISIBLE_DEVICES=${GPU_ID}

TOTAL_TASKS=${#TASKS[@]}
COMPLETED_TASKS=0
FAILED_TASKS=0

echo -e "${MAGENTA}============================================${NC}"
echo -e "${MAGENTA}  开始批量测试${NC}"
echo -e "${MAGENTA}============================================${NC}"
echo ""

for task in "${TASKS[@]}"; do
    COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
    
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  任务 [${COMPLETED_TASKS}/${TOTAL_TASKS}]: ${task}${NC}"
    echo -e "${CYAN}========================================${NC}"
    
    CKPT_PATH="${DOWNLOAD_DIR}/${task}${CHECKPOINT_SUBDIR_SUFFIX}/${CHECKPOINT_FILE}"
    
    # 检查任务环境是否存在
    if [ ! -f "./envs/${task}.py" ]; then
        echo -e "${RED}  ✗ 任务环境不存在: ./envs/${task}.py${NC}"
        echo "${task}: FAILED (环境不存在)" >> "${SUMMARY_FILE}"
        FAILED_TASKS=$((FAILED_TASKS + 1))
        echo ""
        continue
    fi
    
    # 检查任务配置文件
    TASK_CONFIG_FILE="./task_config/${TASK_CONFIG}.yml"
    if [ ! -f "${TASK_CONFIG_FILE}" ]; then
        echo -e "${RED}  ✗ 配置文件不存在: ${TASK_CONFIG_FILE}${NC}"
        echo "${task}: FAILED (配置不存在)" >> "${SUMMARY_FILE}"
        FAILED_TASKS=$((FAILED_TASKS + 1))
        echo ""
        continue
    fi
    
    echo -e "${BLUE}  开始评估...${NC}"
    echo ""
    
    # 运行评估
    PYTHONWARNINGS=ignore::UserWarning \
    python3 << EOF
import sys
import yaml

sys.path.append("./")

# 读取配置
with open("policy/DP/deploy_policy.yml", "r") as f:
    config = yaml.safe_load(f)

# 更新配置
config.update({
    'task_name': '${task}',
    'task_config': '${TASK_CONFIG}',
    'ckpt_setting': 'downloaded_${HF_REPO//\//_}',
    'expert_data_num': '${EXPERT_DATA_NUM}',
    'seed': int('${SEED}'),
    'checkpoint_num': 500,
    'custom_ckpt_path': '${CKPT_PATH}'
})

print(f"配置: task={config['task_name']}, config={config['task_config']}")
print(f"Checkpoint: {config['custom_ckpt_path']}")
print("")

# 运行评估
try:
    from script.eval_policy import main
    main(config)
    exit(0)
except Exception as e:
    print(f"\n✗ 评估失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}  ✓ ${task} 评估完成${NC}"
        
        # 查找最新的结果文件
        RESULT_FILE=$(find "./eval_result/${task}/DP/${TASK_CONFIG}/downloaded_${HF_REPO//\//_}" -name "_result.txt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
        
        if [ -f "${RESULT_FILE}" ]; then
            SUCCESS_RATE=$(tail -1 "${RESULT_FILE}")
            echo "${task}: ${SUCCESS_RATE}" >> "${SUMMARY_FILE}"
            echo -e "${GREEN}  成功率: ${SUCCESS_RATE}${NC}"
        else
            echo "${task}: COMPLETED (未找到结果文件)" >> "${SUMMARY_FILE}"
        fi
    else
        echo ""
        echo -e "${RED}  ✗ ${task} 评估失败${NC}"
        echo "${task}: FAILED (运行时错误)" >> "${SUMMARY_FILE}"
        FAILED_TASKS=$((FAILED_TASKS + 1))
    fi
    
    echo ""
done

# 输出总结
echo -e "${MAGENTA}============================================${NC}"
echo -e "${MAGENTA}  测试完成${NC}"
echo -e "${MAGENTA}============================================${NC}"
echo ""
echo -e "${CYAN}总任务数:${NC} ${TOTAL_TASKS}"
echo -e "${GREEN}成功:${NC} $((TOTAL_TASKS - FAILED_TASKS))"
if [ $FAILED_TASKS -gt 0 ]; then
    echo -e "${RED}失败:${NC} ${FAILED_TASKS}"
fi
echo ""

echo "" >> "${SUMMARY_FILE}"
echo "======================================" >> "${SUMMARY_FILE}"
echo "总任务数: ${TOTAL_TASKS}" >> "${SUMMARY_FILE}"
echo "成功: $((TOTAL_TASKS - FAILED_TASKS))" >> "${SUMMARY_FILE}"
echo "失败: ${FAILED_TASKS}" >> "${SUMMARY_FILE}"

echo -e "${CYAN}结果摘要已保存到:${NC} ${SUMMARY_FILE}"
echo ""

# 显示结果摘要
echo -e "${BLUE}========== 结果摘要 ==========${NC}"
cat "${SUMMARY_FILE}" | grep -A 100 "======"
echo ""

if [ $FAILED_TASKS -eq 0 ]; then
    echo -e "${GREEN}🎉 所有任务测试成功！${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  部分任务失败，请检查日志${NC}"
    exit 1
fi

