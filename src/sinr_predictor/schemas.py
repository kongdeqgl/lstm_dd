"""接口数据结构和严格校验；缺省数值与真正观测到的零值分别记录。"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

UINT64_MAX = 2**64 - 1
FLOAT32_MAX = 3.4028234663852886e38
NODE_NUMERIC_FIELDS = (
    "latitude_deg", "longitude_deg", "altitude_m",
    "velocity_north_mps", "velocity_east_mps", "velocity_down_mps",
    "accel_north_mps2", "accel_east_mps2", "accel_down_mps2",
    "pitch_deg", "yaw_deg", "roll_deg",
)
REQUIRED_NODE_FIELDS = frozenset(NODE_NUMERIC_FIELDS[:6])


class RequestStatus(IntEnum):
    SUCCESS = 0
    INVALID_REQUEST = 1
    MODEL_UNAVAILABLE = 2


class ResultStatus(IntEnum):
    SUCCESS = 0
    INSUFFICIENT_HISTORY = 2
    STALE_OBSERVATION = 3
    UNAVAILABLE = 4


def _object(value: Any, allowed: set[str], required: set[str], name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是对象")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} 的字段名必须是字符串")
    missing = required - value.keys()
    extra = value.keys() - allowed
    if missing or extra:
        raise ValueError(f"{name} 缺少字段 {sorted(missing)}，未知字段 {sorted(extra)}")
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str) or not value.strip()
        or len(value.encode("utf-8")) > 36
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(f"{name} 必须是 1~36 字节的非空标识")
    return value


def _timestamp(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} 必须是 UINT64 毫秒时间戳")
    return value


def _number(value: Any, name: str, bound: float = FLOAT32_MAX) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数值")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} 数值超出范围") from error
    if not math.isfinite(number) or abs(number) > bound:
        raise ValueError(f"{name} 数值超出范围或不是有限数值")
    return number


@dataclass(frozen=True, slots=True)
class NodeRecord:
    node_id: str
    observation_time_ms: int
    member_present: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    velocity_north_mps: float
    velocity_east_mps: float
    velocity_down_mps: float
    accel_north_mps2: float
    accel_east_mps2: float
    accel_down_mps2: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    provided_fields: frozenset[str]

    @classmethod
    def from_dict(
        cls, value: Any, *, default_observation_time_ms: int | None = None
    ) -> "NodeRecord":
        required = {"node_id"}
        if default_observation_time_ms is None:
            required.add("observation_time_ms")
        row = _object(
            value,
            required | {"observation_time_ms", "member_present"} | set(NODE_NUMERIC_FIELDS),
            required,
            "node_record",
        )
        member = row.get("member_present", 1)
        if type(member) is not int or member not in (0, 1):
            raise ValueError("member_present 必须为整数 0 或 1")
        numbers = {
            name: _number(
                row.get(name, 0), name,
                90.0 if name == "latitude_deg" else (
                    180.0 if name == "longitude_deg" else FLOAT32_MAX
                ),
            )
            for name in NODE_NUMERIC_FIELDS
        }
        return cls(
            node_id=_identifier(row["node_id"], "node_id"),
            observation_time_ms=_timestamp(
                row.get("observation_time_ms", default_observation_time_ms),
                "observation_time_ms",
            ),
            member_present=member,
            **numbers,
            provided_fields=frozenset(row.keys() & set(NODE_NUMERIC_FIELDS)),
        )


@dataclass(frozen=True, slots=True)
class RelationRecord:
    source_id: str
    destination_id: str
    observation_time_ms: int
    link_quality: float

    @property
    def key(self) -> tuple[str, str]:
        return self.source_id, self.destination_id

    @classmethod
    def from_dict(
        cls, value: Any, *, default_observation_time_ms: int | None = None
    ) -> "RelationRecord":
        required = {"source_id", "destination_id", "link_quality"}
        if default_observation_time_ms is None:
            required.add("observation_time_ms")
        row = _object(
            value, required | {"observation_time_ms"}, required, "relation_record"
        )
        return cls(
            _identifier(row["source_id"], "source_id"),
            _identifier(row["destination_id"], "destination_id"),
            _timestamp(
                row.get("observation_time_ms", default_observation_time_ms),
                "observation_time_ms",
            ),
            _number(row["link_quality"], "link_quality"),
        )


@dataclass(frozen=True, slots=True)
class QueryRelation:
    source_id: str
    destination_id: str

    @property
    def key(self) -> tuple[str, str]:
        return self.source_id, self.destination_id

    @classmethod
    def from_dict(cls, value: Any) -> "QueryRelation":
        required = {"source_id", "destination_id"}
        row = _object(value, required, required, "query_relation")
        return cls(
            _identifier(row["source_id"], "source_id"),
            _identifier(row["destination_id"], "destination_id"),
        )


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    request_id: str
    node_records: tuple[NodeRecord, ...]
    relation_records: tuple[RelationRecord, ...]
    query_relations: tuple[QueryRelation, ...]

    @classmethod
    def from_dict(
        cls, value: Any, *, default_observation_time_ms: int | None = None
    ) -> "PredictionRequest":
        required = {"request_id", "node_records", "relation_records", "query_relations"}
        row = _object(value, required, required, "request")
        for name in ("node_records", "relation_records", "query_relations"):
            if not isinstance(row[name], list):
                raise ValueError(f"{name} 必须是数组")
        return cls(
            _identifier(row["request_id"], "request_id"),
            tuple(
                NodeRecord.from_dict(
                    item, default_observation_time_ms=default_observation_time_ms
                )
                for item in row["node_records"]
            ),
            tuple(
                RelationRecord.from_dict(
                    item, default_observation_time_ms=default_observation_time_ms
                )
                for item in row["relation_records"]
            ),
            tuple(QueryRelation.from_dict(item) for item in row["query_relations"]),
        )


@dataclass(frozen=True, slots=True)
class PredictionResult:
    source_id: str
    destination_id: str
    target_time_ms: int
    predicted_value: float | None
    status_code: ResultStatus

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "destination_id": self.destination_id,
            "target_time_ms": self.target_time_ms,
            "predicted_value": self.predicted_value,
            "status_code": int(self.status_code),
        }


@dataclass(frozen=True, slots=True)
class PredictionResponse:
    request_id: str
    reference_time_ms: int
    generated_time_ms: int
    status_code: RequestStatus
    results: tuple[PredictionResult, ...] = ()
    # 仅供进程内诊断，不增加外部接口字段。
    diagnostic: str = ""

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "reference_time_ms": self.reference_time_ms,
            "generated_time_ms": self.generated_time_ms,
            "status_code": int(self.status_code),
            "results": [result.to_dict() for result in self.results],
        }
