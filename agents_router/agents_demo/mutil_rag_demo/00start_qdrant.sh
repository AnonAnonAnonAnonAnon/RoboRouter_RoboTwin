#!/bin/bash
# Qdrant 向量数据库启动脚本

echo "=========================================="
echo "  启动 Qdrant 向量数据库"
echo "=========================================="
echo ""

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker未安装"
    echo "请先安装Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查Docker是否运行
if ! docker info &> /dev/null; then
    echo "❌ 错误: Docker未运行"
    echo "请启动Docker服务"
    exit 1
fi

echo "✓ Docker已安装并运行"
echo ""

# 检查Qdrant容器是否已存在
CONTAINER_NAME="qdrant_rb"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "发现已存在的Qdrant容器: ${CONTAINER_NAME}"
    
    # 检查容器状态
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "✓ 容器正在运行"
        echo ""
        echo "Qdrant服务信息:"
        echo "  - HTTP API: http://localhost:6333"
        echo "  - Dashboard: http://localhost:6333/dashboard"
        echo "  - gRPC: localhost:6334"
        echo ""
        echo "如需重启，请先运行: docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}"
        exit 0
    else
        echo "容器已停止，正在启动..."
        docker start ${CONTAINER_NAME}
        if [ $? -eq 0 ]; then
            echo "✓ 容器启动成功"
        else
            echo "❌ 容器启动失败"
            exit 1
        fi
    fi
else
    echo "创建新的Qdrant容器..."
    echo ""
    
    # 检查本地镜像压缩包
    LOCAL_IMAGE_TAR="./scripts/qdrant-v1.15.5.tar"
    if [ -f "${LOCAL_IMAGE_TAR}" ]; then
        echo "发现本地镜像: ${LOCAL_IMAGE_TAR}"
        echo "正在加载本地镜像..."
        docker load -i ${LOCAL_IMAGE_TAR}
        if [ $? -eq 0 ]; then
            echo "✓ 本地镜像加载成功"
        else
            echo "❌ 本地镜像加载失败"
            exit 1
        fi
    else
        echo "未找到本地镜像，将从Docker Hub拉取..."
    fi
    
    # 创建数据目录
    DATA_DIR="./qdrant_storage"
    mkdir -p ${DATA_DIR}
    echo "✓ 数据目录: ${DATA_DIR}"
    
    # 启动Qdrant容器（使用v1.15.5版本）
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p 6333:6333 \
        -p 6334:6334 \
        -v $(pwd)/${DATA_DIR}:/qdrant/storage:z \
        qdrant/qdrant:v1.15.5
    
    if [ $? -eq 0 ]; then
        echo "✓ Qdrant容器创建并启动成功"
    else
        echo "❌ Qdrant容器启动失败"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "  Qdrant 启动成功！"
echo "=========================================="
echo ""
echo "服务信息:"
echo "  - HTTP API:  http://localhost:6333"
echo "  - Dashboard: http://localhost:6333/dashboard"
echo "  - gRPC:      localhost:6334"
echo ""
echo "常用命令:"
echo "  - 查看日志:  docker logs ${CONTAINER_NAME}"
echo "  - 停止服务:  docker stop ${CONTAINER_NAME}"
echo "  - 删除容器:  docker rm ${CONTAINER_NAME}"
echo ""
echo "现在可以运行测试:"
echo "  python test_modules.py"
echo "  python ma_router_retriever_multimodal_qdrant.py"
echo ""

