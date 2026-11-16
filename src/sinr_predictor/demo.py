"""合成数据仅用于验证接口流程，不能用于证明预测精度。"""

from .config import ModelConfig

DEMO_REFERENCE_TIME_MS = 1_700_000_010_000


def make_demo_request(reference_time_ms: int, config: ModelConfig) -> dict:
    nodes = []
    relations = []
    for step in range(config.history_steps):
        timestamp = reference_time_ms - (config.history_steps - 1 - step) * config.sampling_interval_ms
        for offset, node_id in enumerate(("A", "B", "C")):
            nodes.append({
                "node_id": node_id,
                "observation_time_ms": timestamp,
                "member_present": 1,
                "latitude_deg": 31.0,
                "longitude_deg": 121.0 + offset * 0.001,
                "altitude_m": 100.0,
                "velocity_north_mps": 10.0,
                "velocity_east_mps": 0.0,
                "velocity_down_mps": 0.0,
            })
        relations.append({
            "source_id": "A", "destination_id": "B",
            "observation_time_ms": timestamp,
            "link_quality": 10.0 + step * 0.5,
        })
    # 反向链路只有一条陈旧观测；不能复用 A->B 的历史。
    relations.append({
        "source_id": "B", "destination_id": "A",
        "observation_time_ms": reference_time_ms - (config.history_steps - 1) * config.sampling_interval_ms,
        "link_quality": 4.0,
    })
    return {
        "request_id": "demo-001",
        "node_records": nodes,
        "relation_records": relations,
        "query_relations": [
            {"source_id": "A", "destination_id": "B"},
            {"source_id": "B", "destination_id": "A"},
            {"source_id": "A", "destination_id": "C"},
            {"source_id": "A", "destination_id": "UNKNOWN"},
        ],
    }
