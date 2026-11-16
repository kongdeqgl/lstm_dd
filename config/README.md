# 配置文件说明

当前目录只保留两类有效配置：

- `formal_graph_training.toml`：正式图网络 + LSTM 训练配置，包含数据路径、窗口、图层、LSTM 和优化参数。
- `training_parameter_plan_v1.toml`：算料场景生成参数，不参与模型训练。

正式训练命令：

```powershell
.\.venv\Scripts\python.exe -m sinr_predictor.training.graph_train `
  --training-config config\formal_graph_training.toml
```
