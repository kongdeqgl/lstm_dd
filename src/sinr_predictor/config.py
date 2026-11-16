"""模型配置；离线回放和在线预测复用同一组窗口与有效性参数。"""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_version: str = "persistence-v1"
    target_metric_code: int = 1
    sampling_interval_ms: int = 1000
    history_steps: int = 10
    forecast_steps: int = 3
    max_node_age_ms: int = 2000
    max_link_age_ms: int = 2000
    min_observation_ratio: float = 0.8
    max_consecutive_missing_steps: int = 2
    continuity_gap_ms: int = 5000
    cache_retention_ms: int = 600_000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_version, str)
            or not 1 <= len(self.model_version.encode("utf-8")) <= 32
            or any(ord(c) < 32 for c in self.model_version)
        ):
            raise ValueError("model_version 必须是 1~32 字节的非控制字符字符串")
        for name in (
            "sampling_interval_ms", "history_steps", "forecast_steps",
            "continuity_gap_ms", "cache_retention_ms",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        for name in (
            "target_metric_code", "max_node_age_ms", "max_link_age_ms",
            "max_consecutive_missing_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if self.target_metric_code > 65535 or self.sampling_interval_ms > 2**32 - 1:
            raise ValueError("指标编码或采样间隔超出接口整数范围")
        ratio = self.min_observation_ratio
        if (
            isinstance(ratio, bool) or not isinstance(ratio, (int, float))
            or not 0 < ratio <= 1 or not math.isfinite(ratio)
        ):
            raise ValueError("min_observation_ratio 必须在 (0, 1] 内")
        minimum_retention = (
            (self.history_steps - 1) * self.sampling_interval_ms
            + max(self.max_node_age_ms, self.max_link_age_ms, self.continuity_gap_ms)
        )
        if self.cache_retention_ms < minimum_retention:
            raise ValueError("缓存保留时长不足以覆盖历史窗口及有效性检查")

    @classmethod
    def from_toml(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("rb") as stream:
            values = tomllib.load(stream)
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"未知配置项：{', '.join(sorted(unknown))}")
        return cls(**values)

    def model_parameters(self) -> dict[str, str | int]:
        """外部接口文档表 1 中的模型与运行参数。"""
        return {
            "model_version": self.model_version,
            "target_metric_code": self.target_metric_code,
            "sampling_interval_ms": self.sampling_interval_ms,
        }
