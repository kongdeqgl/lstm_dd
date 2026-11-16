# 图空间编码与链路级 LSTM 算法

当前核心实现位于 `models/graph_lstm.py`，训练、监督数据和推理后端分别位于 `training/graph_train.py`、`data/dataset.py`、`models/neural_backend.py`。运行环境为 Python 3.12，计算框架为 PyTorch；图聚合直接使用张量索引和 `index_add_`，不额外依赖 PyG。

## 网络计算

每个历史图的节点原始连续特征为 15 维，有向边为 8 维。逐维标准化后追加同维有效掩码、归一化观测年龄、新观测标记，最终节点输入为 32 维、边输入为 18 维。

节点和边分别通过共享 MLP 编码。对于每条有向边 `u→v`，消息网络读取 `[h_u, h_v, e_uv]`；按接收端汇总消息并除以入度，再通过节点更新网络、残差连接和 LayerNorm 更新节点。无入邻居的节点以零聚合向量更新。图层不使用节点 ID 数值或固定行号。

每条查询链路读取发送端表示、接收端表示和对应关系表示，生成固定长度链路向量，并追加当前历史点是否可用的标记。当前连续运行区间之前的查询输入为零向量且标记为零，不接续退出前的旧历史。

最近 K 个点组织成 `[Q, K, link_dim + 1]`，送入共享参数的单向 LSTM。每次前向从零隐藏状态和记忆状态开始，各查询处于独立批次行。读取最后一层最终隐藏状态，经无输出激活的线性层直接得到 `[Q, H]`。不会把预测值回灌为下一步输入；预测结果允许为负，并反标准化到 dB。

同一时刻的图只编码一次供该请求所有查询复用。多个独立簇历史样本在训练 mini-batch 内分别前向，按有效监督目标数量合并损失后执行一次反向，避免把不同图的节点行号误当成同一身份。

目前提供三种模式，保持相同的 LSTM 和输出定义：

| 模式 | 空间部分 |
| --- | --- |
| graph | 端点与关系编码，加多轮有向图消息聚合 |
| endpoints | 端点与关系编码，不进行邻居聚合 |
| quality | 只读取 SINR 标准化值、其掩码、年龄和新观测标记 |

## 因果样本与监督

训练与在线共享 `HistoryCache`、`build_snapshots`、`build_query_window` 和资格判定。输入只能使用截止前收到且观测时间符合当前网格条件的记录。完整未来日志在独立真值缓存中，仅用于构造监督标签，不传入输入构建路径。

先按 `episode_id` 分组划分训练、验证和测试任务，再在各组生成滑动窗口。同一任务的多个簇必须使用相同 `episode_id`，从而始终进入同一分区；禁止在重叠滑窗生成后随机切分。

默认历史网格从该任务最早观测时间起按采样间隔建立。首个默认基准时刻需覆盖 K 个历史点，最后一个默认时刻预留 H 个未来点；可使用 `reference_times_ms` 显式给出基准时刻。

监督标签当前采用**目标时间上确有观测的实测 SINR**，不对缺失标签插值或保持。目标时刻节点退出、过期，或已经进入基准时间之后的新连续区间，则该监督位置无效。真实日志若不在统一网格上，应先明确目标测量和重采样定义，不能直接把输入保持值当成真实标签。

未显式指定业务查询时，只依据当前截止前已经观测到的有向关系产生查询，不读取未来出现的关系集合。显式指定的查询可以没有历史，届时由资格判定返回历史不足。

## 标准化与损失

统计量只从训练分区的有效字段拟合，验证、测试和在线均复用。原始坐标使用 Python 双精度先做坐标转换和标准化，再转成 float32，避免先将百万米量级 ECEF 坐标转换成 float32 而损失小尺度运动变化。

重叠窗口中的节点新观测和目标标签去重统计；保持的 SINR 不重复计入新测量统计。缺失可选特征统计量采用均值 0、尺度 1、数量 0，并保留掩码。目标 SINR 在所有链路间使用同一组训练统计量。

训练目标为所有有效预测位置上的均方误差，默认各未来步等权，按有效目标数量平均。先索引有效标签再做减法，避免被掩盖的 NaN 污染梯度。无有效标签的样本不参与监督更新。使用 AdamW、梯度范数裁剪和基于验证集整体 RMSE 的早停，恢复最佳验证轮次后再评价独立测试任务。

目前损失对全部有效链路目标平均，因此大簇提供的监督数量较多。如正式实验需要按飞行任务等权，可在既有 mini-batch 损失汇总处增加任务权重，需另行比较其效果。

## 训练日志格式

JSON 顶层为 `episodes` 数组。下面展示一条接收帧的字段，实际训练每个任务需有足够多的连续帧和未来测量，并至少提供三个独立任务供划分。

```json
{
  "episodes": [
    {
      "episode_id": "flight-001",
      "cluster_id": "cluster-A",
      "query_relations": [{"source_id": "A", "destination_id": "B"}],
      "frames": [
        {
          "received_time_ms": 1700000000000,
          "node_records": [
            {
              "node_id": "A", "observation_time_ms": 1700000000000,
              "latitude_deg": 31.0, "longitude_deg": 121.0, "altitude_m": 100.0,
              "velocity_north_mps": 10.0, "velocity_east_mps": 0.0, "velocity_down_mps": 0.0
            },
            {
              "node_id": "B", "observation_time_ms": 1700000000000,
              "latitude_deg": 31.0, "longitude_deg": 121.001, "altitude_m": 100.0,
              "velocity_north_mps": 10.0, "velocity_east_mps": 0.0, "velocity_down_mps": 0.0
            }
          ],
          "relation_records": [
            {"source_id": "A", "destination_id": "B", "observation_time_ms": 1700000000000, "link_quality": 12.5}
          ]
        }
      ]
    }
  ]
}
```

接收帧可以乱序输入，程序按接收时间回放；同一帧中的所有记录以该帧接收时间作为到达时间。观测时间不得晚于接收时间。`query_relations` 和 `reference_times_ms` 均可省略。节点和链路字段沿用在线接口校验规则。

在簇头实时入口中，单时隙新记录可省略 `observation_time_ms`，`Predictor` 以本次请求的服务端接收时间补齐。显式观测时间仍优先于默认值。该便利不应用于离线训练、日志回放或单请求批量补送多个时隙；这些数据必须保留每条记录所属的明确时间。

### 算料生成 1.5 转换

`scripts/data_generation/convert_material_graph_exports.py` 从每个完整运行的 `export/manifest.json` 开始，校验 CSV 分卷 SHA-256、场景快照、时隙连续性、节点和链路唯一性。无效链路不作为已观测边写入，但仍保留对应查询，由历史和标签掩码表示缺测。

仿真器的 x/y 是以固定地球原点沿大圆展开的东/北坐标，z 是高度。转换器复用相同的球面几何得到经纬高，并把速度、加速度映射为当地 NED；图模型后续仍通过公共预处理转为 ECEF。方位角写入 `yaw_deg`，俯仰和滚转保持原单位和含义。

数据按运行保存为分片，`index.json` 保存相对路径和 SHA-256。训练入口既接受原单 JSON，也接受该索引或其所在目录。同一场景族的重复运行共用 `episode_id`，各自的 `scene_id` 作为 `cluster_id`，因此任务划分不会发生场景族泄漏。

### 统一训练配置

`config/formal_graph_training.toml` 以 `[paths]`、`[model]`、`[network]`、`[training]` 四个区段管理一次可复现的训练任务。其中包含数据和输出路径、因果窗口、有效性阈值、图层数、节点/链路/LSTM 维度、优化器及采样间隔。路径相对 TOML 所在目录解析，避免从不同工作目录启动时指向不同数据。

```powershell
.\.venv\Scripts\python.exe -m sinr_predictor.training.graph_train `
  --training-config config\formal_graph_training.toml
```

命令行中明确提供的参数优先于 TOML，可用于临时 smoke 测试。报告会保存实际数据路径、配置路径、窗口、网络和训练参数，而不是只保存 TOML 文本。

## 模型包与评估

`model.pt` 保存格式版本、结构参数、运行与资格参数、固定特征顺序、训练集标准化统计量、权重和训练摘要。加载时检查特征、单位及参数契约，通过 `weights_only=True` 读取张量和基础数据。在线运行参数必须与模型包一致。

`report.json` 分未来步报告可比较标签数量、MAE、RMSE、Bias 和容差命中率；无数据时为 null，不能填零误差。可用率和未来标签缺失比例单独报告。图 LSTM、最近观测保持、历史均值使用相同资格条件和标签掩码，避免在不同样本范围上直接比较误差。

合成日志仅检验训练和保存加载流程，并非飞行通信仿真器。应在真实的独立任务测试集上完成结构消融、成员动态和部署时延评估后，才形成业务精度结论。

PyTorch 接口参考：[LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)、[index_add_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.index_add_.html)、[torch.load](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)。
