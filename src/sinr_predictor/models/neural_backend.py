"""图 LSTM 的在线适配和版本化模型包；仅此可选路径依赖 PyTorch。"""

import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import torch

from ..config import ModelConfig
from ..data.normalization import Normalization
from ..data.preprocess import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, QueryWindow
from ..data.tensors import tensorize_windows
from .graph_lstm import GraphLSTM, NetworkConfig

BUNDLE_FORMAT_VERSION = 1


class TorchBackend:
    def __init__(
        self,
        model: GraphLSTM,
        normalization: Normalization,
        config: ModelConfig,
        *,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.normalization = normalization
        self.config = config
        if model.forecast_steps != config.forecast_steps:
            raise ValueError("模型输出步数与运行配置不一致")

    def predict(
        self, windows: list[QueryWindow] | tuple[QueryWindow, ...], config: ModelConfig
    ) -> list[list[float]]:
        if config != self.config:
            raise ValueError("运行参数与模型包不一致，不能混用窗口和有效性规则")
        if not windows:
            return []
        self.model.eval()
        with torch.inference_mode():
            prediction = self.model(
                tensorize_windows(windows, self.normalization, config, self.device)
            )
            prediction = (
                prediction * self.normalization.target.scale[0] + self.normalization.target.mean[0]
            )
        return prediction.cpu().tolist()

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "TorchBackend":
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle.get("format_version") != BUNDLE_FORMAT_VERSION:
            raise ValueError("不支持的模型包格式版本")
        if bundle.get("node_feature_names") != list(NODE_FEATURE_NAMES) or bundle.get(
            "edge_feature_names"
        ) != list(EDGE_FEATURE_NAMES):
            raise ValueError("模型包特征顺序与当前代码不一致")
        if bundle.get("target_unit") != "dB" or bundle.get("target_metric_name") != "SINR":
            raise ValueError("模型包的目标指标或单位不一致")
        config = ModelConfig(**bundle["model_config"])
        network = NetworkConfig(**bundle["network_config"])
        normalization = Normalization.from_dict(bundle["normalization"])
        model = GraphLSTM(config.forecast_steps, network)
        model.load_state_dict(bundle["state_dict"], strict=True)
        return cls(model, normalization, config, device=device)


def save_bundle(
    path: str | Path,
    model: GraphLSTM,
    normalization: Normalization,
    config: ModelConfig,
    *,
    training_summary: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    content = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "target_metric_name": "SINR",
        "target_unit": "dB",
        "model_config": asdict(config),
        "network_config": asdict(model.config),
        "normalization": normalization.to_dict(),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "training_summary": training_summary or {},
    }
    try:
        torch.save(content, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
