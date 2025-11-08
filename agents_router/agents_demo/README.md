# Agents Demo

## 多模态RAG检索系统

基于 Qdrant + Voyage AI 的多模态向量检索系统，支持文本+图片查询。

### 快速开始

#### 0 下载docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
docker version

#### 0.5 配置 Docker Hub 加速器
# 1) 配置 Docker Hub 加速器
sudo mkdir -p /etc/docker
cat <<'JSON' | sudo tee /etc/docker/daemon.json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://hub-mirror.c.163.com",
    "https://docker.nju.edu.cn"
  ],
  "max-concurrent-downloads": 1
}
JSON

# 2) 重载并重启 Docker，确认生效
sudo systemctl daemon-reload
sudo systemctl restart docker
docker info | grep -A3 "Registry Mirrors"

# 3) 预拉取（用 IPv4，降低超时概率），然后再启动脚本
docker pull --ipv4 qdrant/qdrant:v1.15.5
bash ./00start_qdrant.sh


#### 1. 启动 Qdrant 数据库

```bash
cd mutil_rag_demo
./00start_qdrant.sh
bash ./00start_qdrant.sh
```

#### 2. 安装依赖

```bash
conda activate robo
cd mutil_rag_demo
pip install -r requirements.txt
```

#### 3. 运行查询

```bash
# 多模态查询（文本+图片）
python ma_router_retriever_multimodal_qdrant.py \
  --query "根据当前场景推荐checkpoint" \
  --image "path/to/image.jpg"

python ma_router_retriever_multimodal_qdrant.py \
  --query "根据当前场景推荐checkpoint" \
  --image "/home/cyt/RoboRouter/RoboRouter_RoboTwin/agents_router/frames_to_push/f_0.jpg"



# 运行测试
python ma_router_retriever_multimodal_qdrant.py
```

#### 4. 查看数据库

```bash
python view_qdrant_db.py
```

### 文件说明

- `ma_router_retriever_multimodal_qdrant.py` - 主程序
- `robot_records_data.py` - 数据配置
- `view_qdrant_db.py` - 查看数据库工具
- `mutil_rag_demo/` - RAG核心模块

### 注意

- 首次运行会自动插入数据（约2-3分钟，Voyage API限速）
- 查询时必须同时提供 `--query` 和 `--image` 参数
