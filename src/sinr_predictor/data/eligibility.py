"""资格判定独立于模型数值计算，状态优先级固定。"""

from ..config import ModelConfig
from ..schemas import REQUIRED_NODE_FIELDS, ResultStatus
from .history import HistoryCache
from .preprocess import QueryWindow


def check_eligibility(
    window: QueryWindow, history: HistoryCache, reference_time_ms: int,
    cutoff_ms: int, config: ModelConfig,
) -> ResultStatus:
    query = window.query
    if query.source_id == query.destination_id:
        return ResultStatus.UNAVAILABLE
    endpoints = [history.latest_node(node_id, reference_time_ms, cutoff_ms) for node_id in query.key]
    if any(
        row is None or not row.member_present or not REQUIRED_NODE_FIELDS <= row.provided_fields
        for row in endpoints
    ):
        return ResultStatus.UNAVAILABLE
    if any(reference_time_ms - row.observation_time_ms > config.max_node_age_ms for row in endpoints):
        return ResultStatus.STALE_OBSERVATION
    if window.session_start_ms is None:
        return ResultStatus.INSUFFICIENT_HISTORY
    relation = history.latest_relation(
        query.key, reference_time_ms, cutoff_ms, not_before_ms=window.session_start_ms,
    )
    if relation is not None and reference_time_ms - relation.observation_time_ms > config.max_link_age_ms:
        return ResultStatus.STALE_OBSERVATION
    if (
        relation is None or window.observation_ratio < config.min_observation_ratio
        or window.longest_missing_run > config.max_consecutive_missing_steps
    ):
        return ResultStatus.INSUFFICIENT_HISTORY
    return ResultStatus.SUCCESS
