#!/bin/bash
# Conda环境设置脚本 - 创建robo环境用于多模态RAG测试

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  创建Conda环境: robo"
echo "=========================================="
echo ""

# 环境名称
ENV_NAME="robo"

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "✗ 错误: conda未安装"
    echo "请先安装Anaconda或Miniconda"
    exit 1
fi

echo "✓ conda已安装"
echo ""

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠ 环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建？(y/n): " response
    if [ "$response" = "y" ]; then
        echo "正在删除旧环境..."
        conda env remove -n ${ENV_NAME} -y
        echo "✓ 旧环境已删除"
    else
        echo "已取消"
        exit 0
    fi
fi

# 创建conda环境
echo "步骤1: 创建conda环境 (Python 3.10)..."
conda create -n ${ENV_NAME} python=3.10 -y
echo "✓ 环境创建成功"

# 激活环境
echo ""
echo "步骤2: 激活环境并安装依赖..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# 配置清华镜像源
echo ""
echo "配置清华镜像源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo "✓ 已配置清华镜像源"

# 安装基础依赖
echo ""
echo "安装基础依赖..."
pip install --upgrade pip

# 安装核心依赖
echo ""
echo "安装核心依赖（使用清华镜像）..."
pip install qdrant-client>=1.7.0
pip install voyageai>=0.2.0
pip install requests>=2.31.0
pip install numpy>=1.24.0
pip install pillow>=10.0.0

echo "✓ 核心依赖安装完成"

# 检查是否成功
echo ""
echo "=========================================="
echo "  验证安装"
echo "=========================================="
echo ""

python -c "import qdrant_client; print('✓ qdrant-client: 已安装')" 2>/dev/null && echo "  qdrant-client 可用" || echo "  ✗ qdrant-client 安装失败"
python -c "import voyageai; print('✓ voyageai: 已安装')" 2>/dev/null && echo "  voyageai 可用" || echo "  ✗ voyageai 安装失败"
python -c "import numpy; print('✓ numpy:', numpy.__version__)" 2>/dev/null || echo "  ✗ numpy 安装失败"
python -c "from PIL import Image; print('✓ pillow: 已安装')" 2>/dev/null || echo "  ✗ pillow 安装失败"
python -c "import requests; print('✓ requests: 已安装')" 2>/dev/null || echo "  ✗ requests 安装失败"

echo ""
echo "=========================================="
echo "  ✓ 环境设置完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  1. 激活环境:"
echo "     conda activate robo"
echo ""
echo "  2. 运行测试:"
echo "     cd /data/work/OliverRen/open_s_proj/RoboRouter_RoboTwin/agents_router/agents_demo/mutil_rag_demo"
echo "     python test_voyage_simple.py"
echo ""
echo "  3. 启动Qdrant:"
echo "     ./00start_qdrant.sh"
echo ""
echo "  4. 初始化数据库:"
echo "     export VOYAGE_API_KEY='pa-tEigYTRrvWMOSB2WIcH4f6FkhzEwF8xZ3kUHbFG4hz9'"
echo "     python 01_setup_database.py"
echo ""
echo "  5. 执行检索:"
echo "     python 02_simple_search.py -q '打开笔记本'"
echo ""

