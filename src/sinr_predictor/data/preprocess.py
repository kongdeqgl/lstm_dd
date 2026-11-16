"""因果窗口和未标准化的图特征；坐标统一到 WGS84 ECEF。"""

import math
from dataclasses import dataclass

from ..config import ModelConfig
from ..schemas import REQUIRED_NODE_FIELDS, NodeRecord, QueryRelation
from .history import HistoryCache

type Vector3 = tuple[float, float, float]

# 顺序固定，后续训练时须与标准化参数一起保存到模型包。
NODE_FEATURE_NAMES = (
    "position_x_m", "position_y_m", "position_z_m",
    "velocity_x_mps", "velocity_y_mps", "velocity_z_mps",
    "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    "pitch_sin", "pitch_cos", "yaw_sin", "yaw_cos", "roll_sin", "roll_cos",
)
EDGE_FEATURE_NAMES = (
    "relative_x_m", "relative_y_m", "relative_z_m",
    "relative_vx_mps", "relative_vy_mps", "relative_vz_mps",
    "distance_m", "link_quality_db",
)


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> Vector3:
    """高度按 WGS84 椭球高解释；正式数据如为海拔，须先做基准转换。"""
    latitude, longitude = math.radians(latitude_deg), math.radians(longitude_deg)
    semi_major = 6_378_137.0
    eccentricity_squared = 6.6943799901413165e-3
    radius = semi_major / math.sqrt(1 - eccentricity_squared * math.sin(latitude) ** 2)
    return (
        (radius + altitude_m) * math.cos(latitude) * math.cos(longitude),
        (radius + altitude_m) * math.cos(latitude) * math.sin(longitude),
        (radius * (1 - eccentricity_squared) + altitude_m) * math.sin(latitude),
    )


def ned_to_ecef(
    north: float, east: float, down: float, latitude_deg: float, longitude_deg: float,
) -> Vector3:
    latitude, longitude = math.radians(latitude_deg), math.radians(longitude_deg)
    slat, clat = math.sin(latitude), math.cos(latitude)
    slon, clon = math.sin(longitude), math.cos(longitude)
    return (
        -slat * clon * north - slon * east - clat * clon * down,
        -slat * slon * north + clon * east - clat * slon * down,
        clat * north - slat * down,
    )


@dataclass(frozen=True, slots=True)
class NodeFeatures:
    node_id: str
    values: tuple[float, ...]
    observed_mask: tuple[bool, ...]
    age_ms: int
    is_new: bool
    session_start_ms: int

    @property
    def position_ecef_m(self) -> Vector3:
        return self.values[0], self.values[1], self.values[2]

    @property
    def velocity_ecef_mps(self) -> Vector3:
        return self.values[3], self.values[4], self.values[5]


@dataclass(frozen=True, slots=True)
class EdgeFeatures:
    source_id: str
    destination_id: str
    values: tuple[float, ...]
    observed_mask: tuple[bool, ...]
    quality_age_ms: int | None
    is_new: bool

    @property
    def link_quality(self) -> float | None:
        return self.values[-1] if self.observed_mask[-1] else None


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    time_ms: int
    nodes: dict[str, NodeFeatures]
    edges: dict[tuple[str, str], EdgeFeatures]


@dataclass(frozen=True, slots=True)
class QueryWindow:
    query: QueryRelation
    snapshots: tuple[GraphSnapshot, ...]
    entries: tuple[EdgeFeatures | None, ...]
    observed_mask: tuple[bool, ...]
    session_start_ms: int | None

    @property
    def observation_ratio(self) -> float:
        return sum(self.observed_mask) / len(self.observed_mask)

    @property
    def longest_missing_run(self) -> int:
        longest = current = 0
        for observed in self.observed_mask:
            current = 0 if observed else current + 1
            longest = max(longest, current)
        return longest


def _node_features(row: NodeRecord, grid_ms: int, start_ms: int, interval_ms: int) -> NodeFeatures:
    position = geodetic_to_ecef(row.latitude_deg, row.longitude_deg, row.altitude_m)
    velocity = ned_to_ecef(
        row.velocity_north_mps, row.velocity_east_mps, row.velocity_down_mps,
        row.latitude_deg, row.longitude_deg,
    )
    accel_present = {
        "accel_north_mps2", "accel_east_mps2", "accel_down_mps2",
    } <= row.provided_fields
    acceleration = ned_to_ecef(
        row.accel_north_mps2, row.accel_east_mps2, row.accel_down_mps2,
        row.latitude_deg, row.longitude_deg,
    ) if accel_present else (0.0, 0.0, 0.0)
    values = list(position + velocity + acceleration)
    masks = [True] * 6 + [accel_present] * 3
    for field in ("pitch_deg", "yaw_deg", "roll_deg"):
        present = field in row.provided_fields
        angle = math.radians(getattr(row, field))
        values.extend((math.sin(angle), math.cos(angle)) if present else (0.0, 0.0))
        masks.extend((present, present))
    age = grid_ms - row.observation_time_ms
    return NodeFeatures(
        row.node_id, tuple(values), tuple(masks), age, age < interval_ms, start_ms,
    )


def build_snapshots(
    history: HistoryCache, queries: tuple[QueryRelation, ...], reference_time_ms: int,
    cutoff_ms: int, config: ModelConfig,
) -> tuple[GraphSnapshot, ...]:
    # 空间边采用“已有有向观测 + 本次查询”，避免默认构造 N*(N-1) 个全连接边。
    candidate_pairs = set(history.relations) | {query.key for query in queries}
    snapshots = []
    for index in range(config.history_steps):
        grid_ms = reference_time_ms - (config.history_steps - 1 - index) * config.sampling_interval_ms
        nodes = {}
        for node_id in sorted(history.nodes):
            row = history.latest_node(node_id, grid_ms, cutoff_ms)
            if (
                row is None or not row.member_present
                or not REQUIRED_NODE_FIELDS <= row.provided_fields
                or grid_ms - row.observation_time_ms > config.max_node_age_ms
            ):
                continue
            start = history.active_since(node_id, grid_ms, cutoff_ms, config.continuity_gap_ms)
            assert start is not None
            nodes[node_id] = _node_features(row, grid_ms, start, config.sampling_interval_ms)
        edges = {}
        for source_id, destination_id in sorted(candidate_pairs):
            if source_id == destination_id or source_id not in nodes or destination_id not in nodes:
                continue
            source, destination = nodes[source_id], nodes[destination_id]
            relative_position = tuple(
                end - start for start, end in zip(
                    source.position_ecef_m, destination.position_ecef_m, strict=True,
                )
            )
            relative_velocity = tuple(
                end - start for start, end in zip(
                    source.velocity_ecef_mps, destination.velocity_ecef_mps, strict=True,
                )
            )
            distance = math.hypot(*relative_position)
            record = history.latest_relation(
                (source_id, destination_id), grid_ms, cutoff_ms,
                not_before_ms=max(source.session_start_ms, destination.session_start_ms),
            )
            age = grid_ms - record.observation_time_ms if record else None
            available = record is not None and age <= config.max_link_age_ms
            quality = record.link_quality if available else 0.0
            edges[(source_id, destination_id)] = EdgeFeatures(
                source_id, destination_id,
                relative_position + relative_velocity + (distance, quality),
                (True,) * 7 + (available,), age,
                available and age < config.sampling_interval_ms,
            )
        snapshots.append(GraphSnapshot(grid_ms, nodes, edges))
    return tuple(snapshots)


def build_query_window(
    history: HistoryCache, snapshots: tuple[GraphSnapshot, ...], query: QueryRelation,
    reference_time_ms: int, cutoff_ms: int, config: ModelConfig,
) -> QueryWindow:
    starts = [
        history.active_since(node_id, reference_time_ms, cutoff_ms, config.continuity_gap_ms)
        for node_id in query.key
    ]
    valid_starts = [value for value in starts if value is not None]
    start = max(valid_starts) if len(valid_starts) == 2 else None
    entries = tuple(
        snapshot.edges.get(query.key) if start is not None and snapshot.time_ms >= start else None
        for snapshot in snapshots
    )
    # 短时保持值可作为输入，但不能重复冒充多个新 SINR 测量点。
    observed = tuple(entry is not None and entry.is_new for entry in entries)
    return QueryWindow(query, snapshots, entries, observed, start)
