"""按稳定身份保存观测发生时间与实际到达时间，支持因果快照。"""

from dataclasses import dataclass

from ..schemas import NodeRecord, PredictionRequest, RelationRecord


@dataclass(frozen=True, slots=True)
class Received[T]:
    record: T
    arrived_at_ms: int


def _visible[T: NodeRecord | RelationRecord](
    series: dict[int, Received[T]], grid_time_ms: int, cutoff_ms: int,
) -> tuple[T, ...]:
    return tuple(
        item.record for timestamp, item in sorted(series.items())
        if timestamp <= grid_time_ms and item.arrived_at_ms <= cutoff_ms
    )


class HistoryCache:
    """缓存属于一个簇和一个 Predictor 实例，不跨簇混用。"""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[int, Received[NodeRecord]]] = {}
        self.relations: dict[tuple[str, str], dict[int, Received[RelationRecord]]] = {}

    def ingest(self, request: PredictionRequest, arrived_at_ms: int) -> None:
        # 先检查整批冲突再写入，避免无效请求部分污染缓存。
        node_batch: dict[tuple[str, int], NodeRecord] = {}
        relation_batch: dict[tuple[tuple[str, str], int], RelationRecord] = {}
        for row in request.node_records:
            key = row.node_id, row.observation_time_ms
            prior = node_batch.get(key)
            cached = self.nodes.get(row.node_id, {}).get(row.observation_time_ms)
            if (prior is not None and prior != row) or (
                cached is not None and cached.record != row
            ):
                raise ValueError(f"节点 {row.node_id} 同一观测时间存在冲突记录")
            node_batch[key] = row
        for row in request.relation_records:
            key = row.key, row.observation_time_ms
            prior = relation_batch.get(key)
            cached = self.relations.get(row.key, {}).get(row.observation_time_ms)
            if (prior is not None and prior != row) or (
                cached is not None and cached.record != row
            ):
                raise ValueError(f"链路 {row.key} 同一观测时间存在冲突记录")
            relation_batch[key] = row
        for (node_id, timestamp), row in node_batch.items():
            self.nodes.setdefault(node_id, {}).setdefault(timestamp, Received(row, arrived_at_ms))
        for (relation_key, timestamp), row in relation_batch.items():
            self.relations.setdefault(relation_key, {}).setdefault(
                timestamp, Received(row, arrived_at_ms)
            )

    def latest_node(self, node_id: str, grid_time_ms: int, cutoff_ms: int) -> NodeRecord | None:
        rows = _visible(self.nodes.get(node_id, {}), grid_time_ms, cutoff_ms)
        return rows[-1] if rows else None

    def latest_relation(
        self, key: tuple[str, str], grid_time_ms: int, cutoff_ms: int,
        *, not_before_ms: int = 0,
    ) -> RelationRecord | None:
        rows = _visible(self.relations.get(key, {}), grid_time_ms, cutoff_ms)
        if not rows or rows[-1].observation_time_ms < not_before_ms:
            return None
        return rows[-1]

    def active_since(
        self, node_id: str, grid_time_ms: int, cutoff_ms: int, continuity_gap_ms: int,
    ) -> int | None:
        """退出再加入、或节点观测长时中断，均开启新的连续运行区间。"""
        rows = _visible(self.nodes.get(node_id, {}), grid_time_ms, cutoff_ms)
        if not rows or not rows[-1].member_present:
            return None
        start: int | None = None
        previous: NodeRecord | None = None
        for row in rows:
            if not row.member_present:
                start = None
            elif (
                start is None or previous is None or not previous.member_present
                or row.observation_time_ms - previous.observation_time_ms > continuity_gap_ms
            ):
                start = row.observation_time_ms
            previous = row
        return start

    def prune(self, reference_time_ms: int, retention_ms: int) -> None:
        threshold = reference_time_ms - retention_ms
        for collection in (self.nodes, self.relations):
            for key in list(collection):
                series = collection[key]
                for timestamp in list(series):
                    if timestamp < threshold:
                        del series[timestamp]
                if not series:
                    del collection[key]
