目前实现了基本的制图流程的demo

输入：
RoboRouter_RoboTwin/agents_router/video_test/from_dataset中的视频
视频来自ctx的hf的数据集，每个数据集中的第一个zip，例如https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/blob/main/dataset/beat_block_hammer/aloha-agilex_clean_50.zip
获得zip后拿出其中的video即可，每个video文件夹有50个视频
现在有5个任务，RoboTwin一共有50个任务

多模态表征提取：
RoboRouter_RoboTwin/agents_router/agents_demo/extract_multimodel_embeding_5_task _5_data.py
从文件夹中拿视频，提取首帧
每个文件夹的50个视频，现在指处理前5个，因为调api比较慢
每个任务有50个视频，处理几个可以调
如果本地部署，提取足够快，可以50个都提取
提取的多模态表征都放在RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache/embeddings

t-sne画图：
RoboRouter_RoboTwin/agents_router/agents_demo/draw_t_sne.py
利用文件夹RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache/embeddings中所有的嵌入画图
画图有很多参数可以调，画出来效果不好可以尝试调


输出：RoboRouter_RoboTwin/agents_router/agents_demo/mm_embed_cache/figures

下一步：画更完整的图，包括RoboTwin的50个任务，每个任务更多的点（可以不用50个，看视觉效果）。
其余内容可以在drawio中画，可以先不用管
注意图的呈现效果，调的较为清晰，能看出几十坨不同颜色的彩色即可
画几个版本的图
版本1是常规的，彩色，每个任务文本标在对应的那坨点周围
版本2相比版本1去除图中任务文本，仅有几十坨的点
版本3都是灰色的点，留下一个接口，指定一些任务，即可令对应的一坨点为红色


