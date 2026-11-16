"""生成独立的小规模合成飞行任务，仅验证流程，不代表真实无线信道。"""

import argparse
import json
import math
import random
from pathlib import Path

from ..data.preprocess import geodetic_to_ecef


def make_synthetic_dataset(*, episodes: int = 6, steps: int = 28, seed: int = 42) -> dict:
    if type(episodes) is not int or type(steps) is not int or episodes < 3 or steps < 8:
        raise ValueError("至少生成 3 个独立任务，每个任务至少 8 个时刻")
    rng = random.Random(seed)
    output = []
    for episode in range(episodes):
        frames = []
        node_count = 3 + episode % 2
        phase = rng.uniform(-0.3, 0.3)
        origin = 1_700_000_000_000 + episode * 100_000
        for step in range(steps):
            timestamp = origin + step * 1000
            nodes = []
            positions = []
            for index in range(node_count):
                latitude = 31.0 + index * 0.0003 + step * 0.00005
                longitude = 121.0 + index * 0.0007 + 0.00003 * math.sin(step * 0.15 + index)
                altitude = 100.0 + 2 * math.sin(step * 0.1 + index)
                nodes.append(
                    {
                        "node_id": f"N{index}",
                        "observation_time_ms": timestamp,
                        "member_present": 1,
                        "latitude_deg": latitude,
                        "longitude_deg": longitude,
                        "altitude_m": altitude,
                        "velocity_north_mps": 5.56,
                        "velocity_east_mps": 0.43 * math.cos(step * 0.15 + index),
                        "velocity_down_mps": -0.2 * math.cos(step * 0.1 + index),
                    }
                )
                positions.append(geodetic_to_ecef(latitude, longitude, altitude))
            relations = []
            for source in range(node_count):
                for destination in range(node_count):
                    if source == destination:
                        continue
                    distance = math.dist(positions[source], positions[destination])
                    quality = (
                        19.0
                        - 0.025 * distance
                        + 1.5 * math.sin(step * 0.15 + phase)
                        + 0.6 * (source - destination)
                        + rng.gauss(0, 0.04)
                    )
                    relations.append(
                        {
                            "source_id": f"N{source}",
                            "destination_id": f"N{destination}",
                            "observation_time_ms": timestamp,
                            "link_quality": quality,
                        }
                    )
            frames.append(
                {
                    "received_time_ms": timestamp,
                    "node_records": nodes,
                    "relation_records": relations,
                }
            )
        output.append(
            {
                "episode_id": f"synthetic-flight-{episode:03d}",
                "cluster_id": "cluster-0",
                "query_relations": [
                    {"source_id": f"N{source}", "destination_id": f"N{destination}"}
                    for source in range(node_count)
                    for destination in range(node_count)
                    if source != destination
                ],
                "frames": frames,
            }
        )
    return {"episodes": output}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成合成 SINR 训练日志，仅用于流程验证")
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic_flights.json"))
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    content = make_synthetic_dataset(episodes=args.episodes, steps=args.steps, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
