"""共享有向图编码 + 链路级单向 LSTM + 直接多步回归。"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..data.tensors import EDGE_INPUT_DIM, NODE_INPUT_DIM, TemporalGraphBatch, TensorGraph


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    hidden_dim: int = 64
    link_dim: int = 64
    lstm_hidden_dim: int = 64
    lstm_layers: int = 1
    graph_layers: int = 2
    dropout: float = 0.0
    spatial_mode: str = "graph"

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "link_dim", "lstm_hidden_dim", "lstm_layers", "graph_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} 必须为正整数")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout 必须在 [0, 1) 范围内")
        if not math.isfinite(self.dropout) or self.spatial_mode not in {
            "graph",
            "endpoints",
            "quality",
        }:
            raise ValueError("无效的 dropout 或空间模块模式")


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim)
    )


class DirectedMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = mlp(3 * hidden_dim, hidden_dim, hidden_dim)
        self.update = mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: Tensor, edges: Tensor, edge_index: Tensor) -> Tensor:
        aggregate = torch.zeros_like(nodes)
        degree = nodes.new_zeros((nodes.shape[0], 1))
        if edges.shape[0]:
            source, destination = edge_index
            messages = self.message(torch.cat((nodes[source], nodes[destination], edges), dim=-1))
            aggregate.index_add_(0, destination, messages)
            degree.index_add_(0, destination, nodes.new_ones((edges.shape[0], 1)))
        # 对入邻居取均值；节点无邻居时聚合为零，与节点排列无关。
        aggregate = aggregate / degree.clamp_min(1)
        return self.norm(nodes + self.update(torch.cat((nodes, aggregate), dim=-1)))


class GraphLSTM(nn.Module):
    def __init__(self, forecast_steps: int, config: NetworkConfig | None = None) -> None:
        super().__init__()
        if type(forecast_steps) is not int or forecast_steps < 1:
            raise ValueError("forecast_steps 必须为正整数")
        self.config = config or NetworkConfig()
        self.forecast_steps = forecast_steps
        hidden, link = self.config.hidden_dim, self.config.link_dim
        if self.config.spatial_mode == "quality":
            # 严格质量历史消融：只读 SINR、其掩码、年龄和新观测标记。
            self.quality_encoder = mlp(4, hidden, link)
        else:
            self.node_encoder = mlp(NODE_INPUT_DIM, hidden, hidden)
            self.edge_encoder = mlp(EDGE_INPUT_DIM, hidden, hidden)
            self.message_layers = (
                nn.ModuleList(DirectedMessageLayer(hidden) for _ in range(self.config.graph_layers))
                if self.config.spatial_mode == "graph"
                else nn.ModuleList()
            )
            self.query_encoder = mlp(3 * hidden, hidden, link)
        self.temporal = nn.LSTM(
            input_size=link + 1,
            hidden_size=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=self.config.dropout if self.config.lstm_layers > 1 else 0.0,
        )
        self.output = nn.Linear(self.config.lstm_hidden_dim, forecast_steps)

    def _encode_graph(self, graph: TensorGraph, num_queries: int) -> Tensor:
        selected = graph.query_valid.nonzero(as_tuple=False).flatten()
        result = graph.nodes.new_zeros((num_queries, self.config.link_dim + 1))
        if self.config.spatial_mode == "quality":
            if selected.numel():
                features = graph.edges[graph.query_edges[selected]][:, [7, 15, 16, 17]]
                encoded = self.quality_encoder(features)
            else:
                return result
        else:
            nodes, edges = self.node_encoder(graph.nodes), self.edge_encoder(graph.edges)
            for layer in self.message_layers:
                nodes = layer(nodes, edges, graph.edge_index)
            if not selected.numel():
                return result
            encoded = self.query_encoder(
                torch.cat(
                    (
                        nodes[graph.query_sources[selected]],
                        nodes[graph.query_destinations[selected]],
                        edges[graph.query_edges[selected]],
                    ),
                    dim=-1,
                )
            )
        encoded = torch.cat((encoded, encoded.new_ones((selected.numel(), 1))), dim=-1)
        return result.index_copy(0, selected, encoded)

    def forward(self, batch: TemporalGraphBatch) -> Tensor:
        if not batch.graphs:
            raise ValueError("历史图序列不能为空")
        sequence = torch.stack(
            [self._encode_graph(graph, batch.num_queries) for graph in batch.graphs], dim=1
        )
        # 不传入上次请求状态：每个窗口的 h0/c0 自动初始化为零，查询是独立批次行。
        _, (hidden, _) = self.temporal(sequence)
        return self.output(hidden[-1])


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor | None:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("预测、标签与掩码形状必须一致")
    if not bool(mask.any()):
        return None
    # 先索引有效位置再做减法，缺失标签即使是 NaN 也不会进入损失及梯度。
    return (prediction[mask] - target[mask]).square().mean()
