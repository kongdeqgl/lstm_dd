"""按独立飞行任务划分数据，再用与在线一致的截止规则构建监督样本。"""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..config import ModelConfig
from ..schemas import UINT64_MAX, PredictionRequest, QueryRelation, ResultStatus
from .eligibility import check_eligibility
from .history import HistoryCache
from .preprocess import QueryWindow, build_query_window, build_snapshots


@dataclass(frozen=True, slots=True)
class ReceivedFrame:
    received_time_ms: int
    request: PredictionRequest


@dataclass(frozen=True, slots=True)
class FlightEpisode:
    episode_id: str
    cluster_id: str
    frames: tuple[ReceivedFrame, ...]
    query_relations: tuple[QueryRelation, ...] | None
    reference_times_ms: tuple[int, ...] | None = None

    @classmethod
    def from_dict(cls, row: dict) -> "FlightEpisode":
        if not isinstance(row, dict) or not {"episode_id", "cluster_id", "frames"} <= row.keys():
            raise ValueError("飞行任务必须包含 episode_id、cluster_id、frames")
        if row.keys() - {
            "episode_id",
            "cluster_id",
            "frames",
            "query_relations",
            "reference_times_ms",
        }:
            raise ValueError("飞行任务包含未知字段")
        for name in ("episode_id", "cluster_id"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError(f"{name} 必须为非空字符串")
        if not isinstance(row["frames"], list) or not row["frames"]:
            raise ValueError("frames 必须是非空数组")
        frames = []
        for index, frame in enumerate(row["frames"]):
            if not isinstance(frame, dict) or set(frame) != {
                "received_time_ms",
                "node_records",
                "relation_records",
            }:
                raise ValueError(
                    "frame 字段必须为 received_time_ms、node_records、relation_records"
                )
            received = frame["received_time_ms"]
            if type(received) is not int or not 0 <= received <= UINT64_MAX:
                raise ValueError("received_time_ms 必须是 UINT64 毫秒时间戳")
            request = PredictionRequest.from_dict(
                {
                    "request_id": f"frame-{index}",
                    "node_records": frame["node_records"],
                    "relation_records": frame["relation_records"],
                    "query_relations": [],
                }
            )
            if any(
                record.observation_time_ms > received
                for record in (
                    *request.node_records,
                    *request.relation_records,
                )
            ):
                raise ValueError("记录的观测时间不能晚于其接收时间")
            if any(
                record.source_id == record.destination_id for record in request.relation_records
            ):
                raise ValueError("训练数据不接受自环链路观测")
            frames.append(ReceivedFrame(received, request))
        raw_queries = row.get("query_relations")
        if raw_queries is not None and not isinstance(raw_queries, list):
            raise ValueError("query_relations 必须为数组")
        queries = (
            tuple(QueryRelation.from_dict(value) for value in raw_queries)
            if raw_queries is not None
            else None
        )
        if queries is not None and len({query.key for query in queries}) != len(queries):
            raise ValueError("同一训练任务的查询链路不能重复")
        references = row.get("reference_times_ms")
        if references is not None:
            if (
                not isinstance(references, list)
                or any(
                    type(value) is not int or not 0 <= value <= UINT64_MAX for value in references
                )
                or len(set(references)) != len(references)
            ):
                raise ValueError("reference_times_ms 必须是不重复的 UINT64 时间数组")
            references = tuple(sorted(references))
        return cls(
            row["episode_id"],
            row["cluster_id"],
            tuple(
                sorted(
                    frames,
                    key=lambda frame: frame.received_time_ms,
                )
            ),
            queries,
            references,
        )


@dataclass(frozen=True, slots=True)
class ForecastSample:
    episode_id: str
    cluster_id: str
    reference_time_ms: int
    windows: tuple[QueryWindow, ...]
    statuses: tuple[ResultStatus, ...]
    targets: tuple[tuple[float, ...], ...]
    target_mask: tuple[tuple[bool, ...], ...]

    @property
    def eligible_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index, status in enumerate(self.statuses) if status == ResultStatus.SUCCESS
        )

    @property
    def eligible_windows(self) -> tuple[QueryWindow, ...]:
        return tuple(self.windows[index] for index in self.eligible_indices)


def _episodes_from_document(content: object, source: Path) -> list[FlightEpisode]:
    if (
        not isinstance(content, dict)
        or set(content) != {"episodes"}
        or not isinstance(content["episodes"], list)
    ):
        raise ValueError(f"{source} 顶层必须是仅包含 episodes 数组的对象")
    return [FlightEpisode.from_dict(row) for row in content["episodes"]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_episodes(path: str | Path) -> list[FlightEpisode]:
    """读取单文件数据集，或读取转换器生成的分片索引。

    分片只改变存储方式；每个分片仍使用与原单文件一致的 episodes 格式。
    """
    source = Path(path).resolve()
    if source.is_dir():
        source = source / "index.json"
    content = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(content, dict) or content.get("schema_version") != (
        "flight_episode_shards_v1"
    ):
        episodes = _episodes_from_document(content, source)
    else:
        if set(content) != {"schema_version", "metadata", "shards"}:
            raise ValueError("分片索引字段必须为 schema_version、metadata、shards")
        if not isinstance(content["metadata"], dict) or not isinstance(content["shards"], list):
            raise ValueError("分片索引的 metadata 必须为对象，shards 必须为数组")
        if not content["shards"]:
            raise ValueError("分片索引不能为空")
        root = source.parent.resolve()
        episodes = []
        seen_files = set()
        for item in content["shards"]:
            if not isinstance(item, dict) or set(item) != {"file", "sha256"}:
                raise ValueError("每个分片必须仅包含 file 和 sha256")
            if (
                not isinstance(item["file"], str)
                or not item["file"]
                or not isinstance(item["sha256"], str)
                or len(item["sha256"]) != 64
            ):
                raise ValueError("分片路径或 SHA-256 无效")
            shard = (root / item["file"]).resolve()
            if not shard.is_relative_to(root):
                raise ValueError(f"分片路径越出数据集目录：{item['file']}")
            if shard in seen_files:
                raise ValueError(f"分片重复：{shard}")
            seen_files.add(shard)
            if not shard.is_file() or _sha256(shard) != item["sha256"].lower():
                raise ValueError(f"分片不存在或 SHA-256 不一致：{shard}")
            document = json.loads(shard.read_text(encoding="utf-8-sig"))
            episodes.extend(_episodes_from_document(document, shard))
    if len({(episode.episode_id, episode.cluster_id) for episode in episodes}) != len(episodes):
        raise ValueError("episode_id 与 cluster_id 的组合必须唯一")
    return episodes


def split_episodes(
    episodes: list[FlightEpisode],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[FlightEpisode], list[FlightEpisode], list[FlightEpisode]]:
    if (
        not 0 < validation_fraction < 1
        or not 0 < test_fraction < 1
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("验证和测试比例须大于 0，且之和小于 1")
    groups = sorted({episode.episode_id for episode in episodes})
    if len(groups) < 3:
        raise ValueError("至少需要三个独立 episode_id，才能隔离训练、验证、测试任务")
    random.Random(seed).shuffle(groups)
    n_validation = max(1, int(len(groups) * validation_fraction))
    n_test = max(1, int(len(groups) * test_fraction))
    if n_validation + n_test >= len(groups):
        raise ValueError("当前划分比例没有给训练集留下独立任务")
    test_ids = set(groups[:n_test])
    validation_ids = set(groups[n_test : n_test + n_validation])
    train = [episode for episode in episodes if episode.episode_id not in test_ids | validation_ids]
    validation = [episode for episode in episodes if episode.episode_id in validation_ids]
    test = [episode for episode in episodes if episode.episode_id in test_ids]
    return train, validation, test


def build_samples(
    episodes: list[FlightEpisode],
    config: ModelConfig,
    *,
    stride_steps: int = 1,
) -> list[ForecastSample]:
    if type(stride_steps) is not int or stride_steps < 1:
        raise ValueError("stride_steps 必须为正整数")
    samples = []
    for episode in episodes:
        # 完整真值缓存仅供构造未来监督，不传入模型输入路径。
        truth = HistoryCache()
        times = []
        for frame in episode.frames:
            truth.ingest(frame.request, frame.received_time_ms)
            times.extend(
                record.observation_time_ms
                for record in (
                    *frame.request.node_records,
                    *frame.request.relation_records,
                )
            )
        if not times:
            continue
        references = episode.reference_times_ms
        if references is None:
            first = min(times) + (config.history_steps - 1) * config.sampling_interval_ms
            last = max(times) - config.forecast_steps * config.sampling_interval_ms
            references = range(first, last + 1, stride_steps * config.sampling_interval_ms)
        history = HistoryCache()
        frame_index = 0
        for reference in references:
            if reference + config.forecast_steps * config.sampling_interval_ms > UINT64_MAX:
                raise ValueError("样本目标时间超出 UINT64 范围")
            while (
                frame_index < len(episode.frames)
                and episode.frames[frame_index].received_time_ms <= reference
            ):
                frame = episode.frames[frame_index]
                history.ingest(frame.request, frame.received_time_ms)
                frame_index += 1
            history.prune(reference, config.cache_retention_ms)
            # 未指定业务查询时，只从截止前已观测的关系产生查询，不窥看未来链路集合。
            queries = (
                episode.query_relations
                if episode.query_relations is not None
                else tuple(QueryRelation(*pair) for pair in sorted(history.relations))
            )
            snapshots = build_snapshots(history, queries, reference, reference, config)
            windows = tuple(
                build_query_window(
                    history,
                    snapshots,
                    query,
                    reference,
                    reference,
                    config,
                )
                for query in queries
            )
            statuses = tuple(
                check_eligibility(
                    window,
                    history,
                    reference,
                    reference,
                    config,
                )
                for window in windows
            )
            targets, masks = [], []
            for query in queries:
                values, available = [], []
                for step in range(1, config.forecast_steps + 1):
                    target = reference + step * config.sampling_interval_ms
                    record = truth.relations.get(query.key, {}).get(target)
                    valid = record is not None
                    for node_id in query.key:
                        node = truth.latest_node(node_id, target, UINT64_MAX)
                        start = truth.active_since(
                            node_id, target, UINT64_MAX, config.continuity_gap_ms
                        )
                        valid = (
                            valid
                            and node is not None
                            and node.member_present == 1
                            and (target - node.observation_time_ms <= config.max_node_age_ms)
                            and start is not None
                            and start <= reference
                        )
                    values.append(record.record.link_quality if valid else 0.0)
                    available.append(bool(valid))
                targets.append(tuple(values))
                masks.append(tuple(available))
            samples.append(
                ForecastSample(
                    episode.episode_id,
                    episode.cluster_id,
                    reference,
                    windows,
                    statuses,
                    tuple(targets),
                    tuple(masks),
                )
            )
    return samples
