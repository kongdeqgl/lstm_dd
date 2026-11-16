# 有向链路 SINR 预测算法

## GitHub 仓库说明

仓库保存源码、配置、文档、接口示例和测试，不包含 `.venv`、缓存、压缩包以及 `outputs/` 中的仿真训练数据、模型和报告。克隆后应使用 Python 3.12 配置环境；正式训练前需将本地训练数据复制到配置指定位置，或按下文步骤生成数据。运行正式模型推理前需另行准备 `model.pt`。项目外部的“算料生成 1.5”软件也需另行准备。

```powershell
git clone git@github.com:kongdeqgl/lstm_dd.git
cd lstm_dd
```

本仓库根目录对应本机的 `lstm_new` 文件夹。以下命令均在仓库根目录执行。

第一次看代码请先打开：[代码文件导航](docs/代码文件导航.md)。源码已按数据生成、公共数据处理、模型、训练、推理分类；该文档列出每个文件的用途、阅读顺序和旧路径对照。

甲方现场训练、推理和故障排查请优先阅读：[甲方现场训练与推理测试手册](docs/甲方现场训练与推理测试手册.md)。

算法结构、伪代码、数据转换和源码位置详见：[算法技术要点与实现流程](docs/算法技术要点与实现流程.md)。

现场快速查命令、状态码和常见错误请看：[个人现场速查卡](docs/个人现场速查卡.md)。

当前主线是 **Python 3.12 的有向图网络 + 链路级 LSTM**：每个时隙构建一张簇内有向图，使用节点状态、链路 SINR 和有效性信息，输入过去 10 个时隙，预测未来 3 个时隙的 SINR。正式训练统一使用带中文注释的 `config/formal_graph_training.toml`。

已支持把“算料生成 1.5”的 `output/<run_id>/export/manifest.json` 批量转换为图训练格式，转换脚本为 `scripts/data_generation/convert_material_graph_exports.py`。它会过滤地球遮挡、端点失活等不可用链路，并阻止不同步长、不同模型版本或完全重复运行混入同一数据集。

正式图数据使用按运行分片的 `outputs/complete_graph_training_v1`。

本项目使用 **Python 3.12**，实现**有向图空间编码 + 链路级单向 LSTM + 直接多步 SINR 回归**。包括因果样本构建、训练集标准化、带缺失标签掩码的训练、验证早停、独立任务测试，以及模型包保存和在线加载。

另保留最近观测保持和历史均值基线，以及第一版的字段校验、双时间历史缓存、成员连续性与结果状态管理。基础预测和数据处理使用标准库；神经网络训练与预测需要可选的 PyTorch、NumPy 依赖。当前完成的是算法实现与合成数据流程验证，尚无真实业务精度结论。

## 开发环境

安装 Python 3.12 64 位，使用 VS Code 打开本文件夹。在 PowerShell 中执行：

```powershell
py -3.12 scripts/setup_env.py
```

如果没有 `py` 启动器，用 Python 3.12 解释器的完整路径执行该脚本。脚本创建项目内 `.venv`，并以 `.pth` 将当前项目 `src` 注册到该环境，支持离线开发；不会修改系统 Python 包。已有完整环境会被复用，版本不匹配时会停止。

在 VS Code 中运行 `Python: Select Interpreter`，选择：

```text
.venv\Scripts\python.exe
```

本目录已配置 Python、Pylance、Python Debugger、Ruff 扩展推荐，以及 F5 示例启动项。Ruff 扩展是开发辅助工具，不是算法运行依赖。

项目也提供 `pyproject.toml`，具备构建依赖的环境可采用标准可编辑安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## 核心算法训练

当前项目环境已安装 CPU 版 PyTorch。新环境先安装算法依赖（GPU 环境应按 PyTorch 官方页面选择对应版本）：

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install "numpy>=1.26,<3"
```

甲方 CPU 机器也可以直接使用项目根目录的 `requirements-lock.txt`。现场无网络时，须提前下载对应 Python 3.12 的 NumPy 和 PyTorch CPU wheel，不能只携带该文本文件。

当前训练统一使用 `config/formal_graph_training.toml`，日志格式见 [算法设计与训练数据格式](docs/algorithm.md)。窗口长度、模型维度和阈值仍需结合真实任务验证。

已增加“算料生成 1.5”到图训练分片的转换器。它将每个时隙的全部节点和有向链路组成一张图，将仿真器本地球面 ENU 坐标转换为外部接口使用的经纬高和 NED 向量，并使用场景族作为 `episode_id` 防止重复运行跨训练/验证/测试分区。

```powershell
.\.venv\Scripts\python.exe scripts\data_generation\convert_material_graph_exports.py `
  --run-root "..\算料生成_v1.5_完整源码_20261022_142101\算料生成_v1.5_完整源码\output_graph_training_v1" `
  --output outputs\complete_graph_training_v1 `
  --smoke-groups 3
```

输出 `index.json` 和按运行分片的 `episodes/*.json`。`smoke_index.json` 从三个独立场景族各选一次运行，可用于快速跑通图训练：

```powershell
.\.venv\Scripts\python.exe -m sinr_predictor.training.graph_train `
  --training-config config\formal_graph_training.toml `
  --data outputs\complete_graph_training_v1\smoke_index.json `
  --output outputs\complete_graph_training_v1\smoke_model `
  --spatial-mode graph --hidden-dim 16 --graph-layers 1 `
  --epochs 1 --patience 1 --stride-steps 50
```

正式训练可使用带逐项中文注释的统一配置 [formal_graph_training.toml](config/formal_graph_training.toml)。它同时保存数据和输出路径、历史/预测窗口、图网络结构、LSTM 结构以及优化参数：

```powershell
.\.venv\Scripts\python.exe -m sinr_predictor.training.graph_train `
  --training-config config\formal_graph_training.toml
```

TOML 中的相对路径以配置文件所在目录为基准。如果临时需要小数据、少轮次实验，仍可在同一命令后追加 `--data`、`--output`、`--epochs` 等参数覆盖 TOML，不用修改正式配置。

消融模式通过 `--spatial-mode` 选择，分别保存到独立输出目录：

| 模式 | 输入与结构 |
| --- | --- |
| `graph`（默认） | 节点/有向关系编码、邻居消息聚合、查询链路表示、共享 LSTM |
| `endpoints` | 端点状态与关系编码、共享 LSTM，不聚合邻居 |
| `quality` | 仅 SINR 历史、对应掩码与年龄、共享 LSTM |

在比较模式时保持相同任务划分、种子、训练预算和 LSTM 维度。默认容差为示例用的 3 dB，可通过 `--tolerance-db` 在正式测试前固定为业务约定值。

加载训练好的模型包进行预测：

```python
from sinr_predictor import Predictor

predictor = Predictor.from_model_bundle(
    "outputs/complete_graph_training_v1/formal_model/model.pt"
)
response = predictor.predict(current_request)
```

也可在 JSON 命令行入口使用正式模型包。模型包自带运行配置，不能再同时指定 `--config`：

```powershell
.\.venv\Scripts\python.exe -m sinr_predictor `
  --request examples\request.json `
  --model outputs\complete_graph_training_v1\formal_model\model.pt `
  --replay-time-ms 1700000010000 `
  --output outputs\prediction.json
```

CLI 一次运行只处理一个请求。连续在线请求应复用下文的同一个 `Predictor` 实例。命令行退出码：0 表示请求处理成功（各链路仍须检查状态），2 表示请求、配置或文件输入无效，3 表示预测后端不可用。

## 进程内调用

```python
import json
from pathlib import Path

from sinr_predictor import Predictor

predictor = Predictor.from_model_bundle(
    "outputs/complete_graph_training_v1/formal_model/model.pt"
)
request = json.loads(Path("your_current_request.json").read_text(encoding="utf-8"))
response = predictor.predict(request)

print(response.to_dict())
print(response.diagnostic)  # 内部诊断，不会写入标准响应字段
```

每个簇使用自己的 `Predictor` 实例，持续复用以保存历史。后续请求可只发送新增观测；数组可以为空。请求时间必须非递减；独立回放任务应创建新的实例。

请求字段与接口文档保持一致：

```json
{
  "request_id": "request-001",
  "node_records": [],
  "relation_records": [],
  "query_relations": [
    {"source_id": "A", "destination_id": "B"}
  ]
}
```

没有可用缓存时，此示例会返回历史不足或端点不可用，而非有效数值。完整合成请求见 `examples/request.json`。

簇头在线部署可只使用簇头接收时间。在 `Predictor.predict()` 入口，节点或链路记录未提供 `observation_time_ms` 时，服务会用本次请求的服务端基准时间自动补齐。若上游将来能提供可靠源端时间，显式值优先且保持兼容。

自动打时间戳只适用于“每个请求携带当前时隙新记录”的在线模式。如果启动时一次补送多个历史时隙，每条历史记录仍须明确填写 `observation_time_ms`；否则同一节点的多条数据会被视为同一接收时间的冲突记录。离线训练和回放数据始终要求显式时间。

响应按**原查询顺序优先、未来步数次之**平铺。目标时间为 `reference_time_ms + step * sampling_interval_ms`，不对基准时间向整秒取整。基准时间在预测函数入口由服务端确定，在线请求不能覆盖它。

请求级状态为 0 成功、1 请求无效、2 模型不可用。结果级状态为 0 成功、2 历史不足、3 观测过期、4 不可用。无效结果的 `predicted_value` 使用 Python `None` / JSON `null`，合法零 SINR 仍返回 `0.0` 且状态为 0。

**当前实现是 Python/JSON 逻辑接口。** 接口文档的固定字节类型尚未对应到 TCP 二进制封包。`null` 对应的 FLOAT32 二进制占位、字节序、传输协议均留待接口约定，不在本版臆定。格式错误到无法识别请求 ID 时，错误响应的 `request_id` 为空字符串。

## 代码分类

### 1. 数据生成与转换

| 文件 | 责任 |
| --- | --- |
| `scripts/data_generation/generate_graph_training_scenarios.py` | 按参数计划生成仿真场景配置 |
| `scripts/data_generation/convert_material_graph_exports.py` | 将算料输出转换成分片图训练数据 |
| `scripts/data_generation/analyze_material_ranges.py` | 检查算料字段范围和异常值 |
| `src/sinr_predictor/data_generation/material_graph.py` | 完整性校验、坐标转换、按时隙构图和索引生成 |
| `src/sinr_predictor/data_generation/synthetic.py` | 生成仅供快速验证的合成数据 |

### 2. 公共数据处理（训练、推理共用）

| 文件 | 责任 |
| --- | --- |
| `src/sinr_predictor/data/history.py` | 历史缓存、去重和成员连续区间 |
| `src/sinr_predictor/data/preprocess.py` | 坐标转换、因果历史图、节点/边特征 |
| `src/sinr_predictor/data/eligibility.py` | 判断链路是否满足预测条件 |
| `src/sinr_predictor/data/dataset.py` | 训练数据读取、任务划分、回放、监督窗口和标签 |
| `src/sinr_predictor/data/normalization.py` | 标准化统计量拟合与应用 |
| `src/sinr_predictor/data/tensors.py` | 变规模历史图和查询链路张量化 |

### 3. 模型与模型包

| 文件 | 责任 |
| --- | --- |
| `src/sinr_predictor/models/graph_lstm.py` | 图消息聚合、查询编码、LSTM 和多步输出 |
| `src/sinr_predictor/models/neural_backend.py` | 模型包保存、加载及 PyTorch 预测 |
| `src/sinr_predictor/models/backends.py` | 统一预测协议、最近观测和历史均值基线 |

### 4. 训练与评估

| 文件 | 责任 |
| --- | --- |
| `src/sinr_predictor/training/graph_train.py` | 训练命令、优化循环、验证早停和导出 |
| `src/sinr_predictor/training/evaluate.py` | 按原始 dB 单位计算评估指标 |

### 5. 在线推理

| 文件 | 责任 |
| --- | --- |
| `src/sinr_predictor/inference/predictor.py` | 单簇历史维护、请求编排、模型调用和结果封装 |
| `src/sinr_predictor/inference/cli.py` | JSON 文件推理命令行入口 |

### 6. 公共接口

`src/sinr_predictor/config.py` 定义公共运行配置；`schemas.py` 定义数据结构、字段校验和状态码。顶层旧转发模块已删除，阅读和修改算法请进入对应功能子目录。训练统一使用 `python -m sinr_predictor.training.graph_train`，推理仍使用 `python -m sinr_predictor`；对外 Python 接口 `from sinr_predictor import Predictor` 保持不变。

## 图模型与毫秒接口的数据处理约定

- 全部时间为同一时基上的毫秒时间戳；SINR 单位为 dB。需要由采集系统保证时钟同步。
- 同时保存观测发生时间和进入预测实例的到达时间。快照只选取发生时间不晚于网格点、到达时间不晚于本次截止的记录，不使用双向插值。
- 同一个观测批次到达时，用请求接收时间作为批内记录的到达时间。在线记录省略 `observation_time_ms` 时，也使用该服务端时间作为有效观测时间。若要回放真实网络延迟，必须按接收日志逐批重放，不能把完整日志一次性装入后声称验证了延迟场景。
- 相同身份和观测时间的完全相同记录幂等；冲突记录使整个请求无效，且该批新记录不会部分写入缓存。缓存按观测时间保留最近 `cache_retention_ms` 范围。
- 节点退出、再加入或超过 `continuity_gap_ms` 后才重新观测到，会开启新连续区间。涉及该节点的链路不能接续旧区间历史。短时间重启若没有退出或中断证据，现有字段无法识别，需后续补充会话标识。
- 本版要求成功预测的端点显式提供经度、纬度、高度和三个 NED 速度分量。省略字段虽按接口默认值解析，仍保留缺测标记，不把默认零冒充测量。加速度和姿态为可选。
- 经纬高按 WGS84 椭球高转换为 ECEF；NED 速度、加速度旋转到 ECEF 后计算相对量。高度若为海拔、或加速度包含特定机体系含义，应先约定并转换。
- 默认空间边为“缓存中存在有向链路观测的端点对 + 本次查询端点对”，过滤不可用端点，不默认构造全连接图。
- 短时保持有独立观测年龄；只有本时间区间内新测得的 SINR 才计入新观测覆盖比例，保持值不会重复贡献多个测量点。
- 节点和链路特征保留物理尺度及掩码；神经网络张量化时使用模型包中的训练集标准化参数，缺失占位不参与拟合。基线不消费运动特征。
- 状态判定优先级为：端点或必要字段不可用 → 必需观测过期 → 历史覆盖不足 → 成功。部分链路失败不使整个请求失败。
- 生成结果时再次检查目标时效。目标时间已经到达的成功候选结果改为不可用，剩余未来步可以保留成功。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖方向隔离、双时间因果性、合法零与缺测、成员重入簇、长时中断、批次原子性、目标失效，以及神经网络梯度、真实张量重排、邻居影响、严格质量消融、变规模图和训练保存加载。无需安装 pytest；未安装 torch 时神经网络测试会跳过。

移动项目后可重新执行 `scripts/setup_env.py` 修复源码注册；开发注册按当前虚拟环境定位 `src`。命令统一使用 `.venv\\Scripts\\python.exe -m ...`，避免移动前生成的命令启动器保留旧路径。

## 真实数据接入与后续验证

1. 按真实飞行任务整理接收日志，确认 SINR 单位、目标时间网格、坐标基准和指标编码，替换合成数据。
2. 完成三种结构的消融实验，并按预测步长、成员规模、成员变化和机动类型分组检查误差与覆盖率。
3. 在实际部署硬件上验证图构建和推理耗时，再确定窗口、图层数和 LSTM 维度；当前未做端侧性能验收。
