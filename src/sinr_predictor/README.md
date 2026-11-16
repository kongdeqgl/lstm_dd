# 源码阅读入口

详细目录树、关键函数和阅读顺序见[代码文件导航](../../docs/代码文件导航.md)。

| 位置 | 功能 |
| --- | --- |
| `config.py`、`schemas.py` | 公共配置、输入输出结构、校验和状态码 |
| `data_generation/` | 仿真算料转换、合成数据生成 |
| `data/` | 历史、构图、有效性、离线样本、标准化、张量 |
| `models/` | 图网络 + LSTM、基线、模型包保存加载 |
| `training/` | 训练编排和评估 |
| `inference/` | 在线推理编排和命令行 |
| `demo.py` | 人工接口示例 |

顶层旧转发模块已删除，每项功能只有分类目录中的实际实现。
训练命令使用 `python -m sinr_predictor.training.graph_train`，推理使用 `python -m sinr_predictor`。
