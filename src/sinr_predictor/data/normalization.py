"""训练集统计量，逐维忽略缺失占位；Python 双精度累计避免 ECEF 消减误差。"""

import math
from dataclasses import asdict, dataclass

from ..config import ModelConfig
from .dataset import ForecastSample
from .preprocess import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES


class _Moments:
    def __init__(self, dimension: int) -> None:
        self.count = [0] * dimension
        self.mean = [0.0] * dimension
        self.m2 = [0.0] * dimension

    def add(self, values, mask) -> None:
        for index, (value, available) in enumerate(zip(values, mask, strict=True)):
            if not available:
                continue
            if not math.isfinite(value):
                raise ValueError("有效特征和标签必须是有限值")
            self.count[index] += 1
            delta = value - self.mean[index]
            self.mean[index] += delta / self.count[index]
            self.m2[index] += delta * (value - self.mean[index])

    def finish(self) -> "FeatureStatistics":
        scales = []
        for count, variance in zip(self.count, self.m2, strict=True):
            std = math.sqrt(max(0.0, variance / count)) if count else 0.0
            scales.append(std if std > 1e-6 else 1.0)
        return FeatureStatistics(tuple(self.count), tuple(self.mean), tuple(scales))


@dataclass(frozen=True, slots=True)
class FeatureStatistics:
    count: tuple[int, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def validate(self, dimension: int) -> None:
        if (
            len(self.count) != dimension
            or len(self.mean) != dimension
            or len(self.scale) != dimension
        ):
            raise ValueError("标准化参数维度与特征顺序不一致")
        if any(type(value) is not int or value < 0 for value in self.count):
            raise ValueError("标准化样本数量必须为非负整数")
        if any(not math.isfinite(value) for value in self.mean) or any(
            not math.isfinite(value) or value <= 0 for value in self.scale
        ):
            raise ValueError("标准化参数必须有限且尺度为正")

    def transform(self, values, mask) -> list[float]:
        return [
            (value - mean) / scale if available else 0.0
            for value, available, mean, scale in zip(
                values, mask, self.mean, self.scale, strict=True
            )
        ]


@dataclass(frozen=True, slots=True)
class Normalization:
    nodes: FeatureStatistics
    edges: FeatureStatistics
    target: FeatureStatistics

    def __post_init__(self) -> None:
        self.nodes.validate(len(NODE_FEATURE_NAMES))
        self.edges.validate(len(EDGE_FEATURE_NAMES))
        self.target.validate(1)
        if not self.target.count[0]:
            raise ValueError("训练集中没有有效监督标签")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Normalization":
        return cls(
            **{
                name: FeatureStatistics(
                    **{field: tuple(values) for field, values in value[name].items()}
                )
                for name in ("nodes", "edges", "target")
            }
        )

    @classmethod
    def fit(cls, samples: list[ForecastSample], config: ModelConfig) -> "Normalization":
        nodes = _Moments(len(NODE_FEATURE_NAMES))
        edges = _Moments(len(EDGE_FEATURE_NAMES))
        targets = _Moments(1)
        seen_nodes, seen_geometry, seen_quality, seen_targets = set(), set(), set(), set()
        for sample in samples:
            indices = sample.eligible_indices
            if not indices:
                continue
            context = sample.episode_id, sample.cluster_id
            for graph in sample.windows[indices[0]].snapshots:
                for node in graph.nodes.values():
                    key = (*context, node.node_id, graph.time_ms - node.age_ms)
                    if node.is_new and key not in seen_nodes:
                        nodes.add(node.values, node.observed_mask)
                        seen_nodes.add(key)
                for key, edge in graph.edges.items():
                    geometry_key = (*context, *key, graph.time_ms)
                    quality_key = (*context, *key, graph.time_ms - (edge.quality_age_ms or 0))
                    mask = list(edge.observed_mask)
                    mask[:7] = [geometry_key not in seen_geometry] * 7
                    mask[-1] = mask[-1] and edge.is_new and quality_key not in seen_quality
                    edges.add(edge.values, mask)
                    seen_geometry.add(geometry_key)
                    if mask[-1]:
                        seen_quality.add(quality_key)
            for index in indices:
                query = sample.windows[index].query
                for step, (value, valid) in enumerate(
                    zip(
                        sample.targets[index],
                        sample.target_mask[index],
                        strict=True,
                    ),
                    start=1,
                ):
                    key = (
                        *context,
                        *query.key,
                        sample.reference_time_ms + step * config.sampling_interval_ms,
                    )
                    if valid and key not in seen_targets:
                        targets.add((value,), (True,))
                        seen_targets.add(key)
        return cls(nodes.finish(), edges.finish(), targets.finish())
