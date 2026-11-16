# 数据生成与转换脚本

当前图模型的数据准备分为三步：

1. `generate_graph_training_scenarios.py`：根据参数计划生成仿真场景 JSON。
2. 在“算料生成 1.5”中运行这些场景，得到各运行目录下的 `export/manifest.json` 和 CSV 分卷。
3. `convert_material_graph_exports.py`：校验算料并转换为图训练分片和索引。

生成场景：

```powershell
.\.venv\Scripts\python.exe scripts\data_generation\generate_graph_training_scenarios.py `
  --config config\training_parameter_plan_v1.toml `
  --output outputs\generated_scenarios
```

转换图数据：

```powershell
.\.venv\Scripts\python.exe scripts\data_generation\convert_material_graph_exports.py `
  --run-root <算料输出目录> `
  --output outputs\complete_graph_training_v1 `
  --smoke-groups 3
```

`analyze_material_ranges.py` 用于检查数值范围。
