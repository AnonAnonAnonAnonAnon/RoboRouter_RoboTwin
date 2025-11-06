# Agents Demo

## 多模态RAG检索系统

基于 Qdrant + Voyage AI 的多模态向量检索系统，支持文本+图片查询。

### 快速开始

#### 1. 启动 Qdrant 数据库

```bash
cd mutil_rag_demo
./00start_qdrant.sh
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
