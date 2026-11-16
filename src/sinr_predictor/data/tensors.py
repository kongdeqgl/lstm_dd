"""将同一请求的变规模历史图转换为张量；不按节点位置关联历史。"""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import ModelConfig
from .normalization import Normalization
from .preprocess import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, QueryWindow

NODE_INPUT_DIM = 2 * len(NODE_FEATURE_NAMES) + 2
EDGE_INPUT_DIM = 2 * len(EDGE_FEATURE_NAMES) + 2


@dataclass(frozen=True, slots=True)
class TensorGraph:
    nodes: Tensor
    edges: Tensor
    edge_index: Tensor
    query_sources: Tensor
    query_destinations: Tensor
    query_edges: Tensor
    query_valid: Tensor


@dataclass(frozen=True, slots=True)
class TemporalGraphBatch:
    graphs: tuple[TensorGraph, ...]
    num_queries: int


def tensorize_windows(
    windows: tuple[QueryWindow, ...] | list[QueryWindow],
    normalization: Normalization,
    config: ModelConfig,
    device: str | torch.device = "cpu",
) -> TemporalGraphBatch:
    if not windows:
        raise ValueError("张量批次必须至少包含一条查询")
    snapshots = windows[0].snapshots
    if len(snapshots) != config.history_steps or any(
        window.snapshots is not snapshots for window in windows
    ):
        raise ValueError("一个张量批次必须来自同一历史图序列，且窗口长度与配置一致")
    graphs = []
    for step, snapshot in enumerate(snapshots):
        node_ids, edge_keys = sorted(snapshot.nodes), sorted(snapshot.edges)
        node_indices = {node_id: index for index, node_id in enumerate(node_ids)}
        edge_indices = {key: index for index, key in enumerate(edge_keys)}
        node_values = []
        for node_id in node_ids:
            node = snapshot.nodes[node_id]
            node_values.append(
                normalization.nodes.transform(node.values, node.observed_mask)
                + [float(value) for value in node.observed_mask]
                + [min(node.age_ms / max(1, config.max_node_age_ms), 10.0), float(node.is_new)]
            )
        edge_values = []
        for key in edge_keys:
            edge = snapshot.edges[key]
            age = (
                10.0
                if edge.quality_age_ms is None
                else min(
                    edge.quality_age_ms / max(1, config.max_link_age_ms),
                    10.0,
                )
            )
            edge_values.append(
                normalization.edges.transform(edge.values, edge.observed_mask)
                + [float(value) for value in edge.observed_mask]
                + [age, float(edge.is_new)]
            )
        sources, destinations, edges, valid = [], [], [], []
        for window in windows:
            key = window.query.key
            available = window.entries[step] is not None and key in edge_indices
            sources.append(node_indices[key[0]] if available else -1)
            destinations.append(node_indices[key[1]] if available else -1)
            edges.append(edge_indices[key] if available else -1)
            valid.append(available)
        graphs.append(
            TensorGraph(
                torch.tensor(node_values, dtype=torch.float32, device=device).reshape(
                    -1, NODE_INPUT_DIM
                ),
                torch.tensor(edge_values, dtype=torch.float32, device=device).reshape(
                    -1, EDGE_INPUT_DIM
                ),
                torch.tensor(
                    [
                        [node_indices[key[0]] for key in edge_keys],
                        [node_indices[key[1]] for key in edge_keys],
                    ],
                    dtype=torch.long,
                    device=device,
                ).reshape(2, -1),
                torch.tensor(sources, dtype=torch.long, device=device),
                torch.tensor(destinations, dtype=torch.long, device=device),
                torch.tensor(edges, dtype=torch.long, device=device),
                torch.tensor(valid, dtype=torch.bool, device=device),
            )
        )
    return TemporalGraphBatch(tuple(graphs), len(windows))
