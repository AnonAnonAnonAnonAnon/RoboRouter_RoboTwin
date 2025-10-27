RoboRouter_RoboTwin/agents文件夹，放agent相关的部分实现

整体上，RoboRouter基于RoboTwin，构建一些agent进行实现

RoboRouter_RoboTwin/agents/ma_router_retriever_min_gpt_demo.py 成功实现agents的调用

RoboRouter_RoboTwin/agents/evaluator_min_gpt_demo.py 试图实现：输入视频，是否成功，任务描述；提取几帧；提交给gpt；输出情况分析，例如如何失败
但是未能成功实现，主要问题是chatanywhere的服务在接收图片上传方面似乎有问题
ChatAnyWhere 网关对 data:、file_id 不可用，对多家外链域抓取失败或被限制（invalid_image_url）；加代理也并未彻底绕过


