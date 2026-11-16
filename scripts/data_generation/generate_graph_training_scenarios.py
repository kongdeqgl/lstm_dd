"""按受控覆盖计划生成可供算料程序校验的GraphLSTM场景参数。"""

from __future__ import annotations

import argparse
import json
import math
import random
import tomllib
from copy import deepcopy
from pathlib import Path


def jammer_frequency(overlap: float) -> tuple[float, float]:
    # 接收频带为2.39~2.41 GHz；干扰带宽固定20 MHz。
    starts = {1.0: 2.39e9, 0.5: 2.4e9, 0.25: 2.405e9}
    start = starts[overlap]
    return start, start + 20e6


def one_jammer(
    profile: str,
    family: int,
    replicate: int,
    duration: float,
    route_distance: float,
    heading_deg: float,
    config: dict,
    index: int = 1,
) -> dict:
    jammer = config["jammer"]
    angle = math.radians(heading_deg)
    along = route_distance * ([0.3, 0.5, 0.7][(family + index) % 3])
    lateral = jammer["lateral_offset_choices_m"][(family + replicate + index) % 3]
    center = [
        along * math.cos(angle) - lateral * math.sin(angle),
        along * math.sin(angle) + lateral * math.cos(angle),
        jammer["altitude_choices_m"][(family + replicate + index) % 3],
    ]
    overlap = jammer["frequency_overlap_choices"][(family + replicate + index) % 3]
    frequency_start, frequency_end = jammer_frequency(overlap)
    if profile == "weak":
        strength, start, end, trajectory = 150.0, 0.1 * duration, 0.9 * duration, "CIRCULAR"
    elif profile == "medium":
        strength, start, end, trajectory = 700.0, 0.2 * duration, 0.8 * duration, "LINEAR"
    elif profile == "strong":
        strength, start, end, trajectory = 1800.0, 0.15 * duration, 0.85 * duration, "CIRCULAR"
    elif profile == "pulse":
        strength, start, end, trajectory = 4000.0, 0.4 * duration, 0.6 * duration, "FIXED"
    elif profile == "dual" and index == 1:
        strength, start, end, trajectory = 700.0, 0.0, 0.65 * duration, "CIRCULAR"
    elif profile == "dual":
        strength, start, end, trajectory = 2200.0, 0.35 * duration, duration, "LINEAR"
    else:
        strength, start, end, trajectory = 0.0, 0.0, duration, "FIXED"
    strength *= (0.9, 0.97, 1.04, 1.12)[replicate]
    result = {
        "jammer_id": f"J{index:02d}",
        "trajectory_type": trajectory,
        "position_m": center,
        "strength_w": strength,
        "frequency_start_hz": frequency_start,
        "frequency_end_hz": frequency_end,
        "effect_start_s": start,
        "effect_end_s": end,
    }
    if trajectory == "CIRCULAR":
        radius = jammer["radius_choices_m"][(family + replicate + index) % 3]
        result.update(
            center_m=center,
            radius_m=radius,
            angular_velocity_radps=jammer["angular_velocity_choices_radps"][
                (family + replicate + index) % 3
            ],
            initial_phase_rad=0.5 * replicate + 0.3 * index,
        )
        result["position_m"] = [center[0] + radius, center[1], center[2]]
    elif trajectory == "LINEAR":
        cross_speed = (180.0, 300.0, 450.0)[(family + replicate + index) % 3]
        result["velocity_mps"] = [
            -cross_speed * math.sin(angle),
            cross_speed * math.cos(angle),
            0.0,
        ]
    return result


def build_scenario(config: dict, family: int, replicate: int) -> dict:
    cluster, motion, simulation = config["cluster"], config["motion"], config["simulation"]
    node_count = cluster["node_count_choices"][family % len(cluster["node_count_choices"])]
    formation = cluster["formation_choices"][(family // 2 + replicate) % 3]
    spacing = cluster["spacing_choices_m"][(family + replicate) % 4]
    base_speed = motion["speed_choices_mps"][(2 * family + replicate) % 5]
    speed = base_speed * (0.96, 1.0, 1.04, 1.08)[replicate]
    duration = simulation["duration_choices_s"][(family + replicate // 2) % 2]
    heading = motion["heading_choices_deg"][(family + replicate) % 7]
    origin_altitude = motion["origin_altitude_choices_m"][(family + replicate) % 4]
    altitude_offset = motion["target_altitude_offset_choices_m"][(family + 2 * replicate) % 3]
    target_altitude = max(10000.0, origin_altitude + altitude_offset)
    route_distance = speed * duration * motion["route_distance_factor"]
    angle = math.radians(heading)
    target = [route_distance * math.cos(angle), route_distance * math.sin(angle), target_altitude]
    profile = config["jammer"]["profile_choices"][(5 * family + family // 6) % 6]
    jammer_count = 2 if profile == "dual" else 1
    jammers = [
        one_jammer(
            profile,
            family,
            replicate,
            duration,
            route_distance,
            heading,
            config,
            index,
        )
        for index in range(1, jammer_count + 1)
    ]
    scene_id = f"graph-v1-f{family + 1:02d}-r{replicate + 1:02d}"
    link = deepcopy(config["link"])
    return {
        "scene_id": scene_id,
        "node_groups": [
            {
                "group_id": "G01",
                "count": node_count,
                "origin_m": [0.0, (replicate - 1.5) * 5000.0, origin_altitude],
                "target_position_m": target,
                "formation": formation,
                "spacing_m": spacing,
                "speed_mps": speed,
                "heading_deg": heading,
                "launch_interval_s": cluster["launch_interval_choices_s"][
                    (family + replicate) % 3
                ],
                "aircraft_type": "HGV",
                "cruise_altitude_m": max(origin_altitude, target_altitude, 30000.0),
            }
        ],
        "jammers": jammers,
        "link": link,
        "tasks": [],
        "simulation": {
            "step_s": simulation["step_s"],
            "duration_s": duration,
            "random_seed": config["random_seed"] + family * 100 + replicate,
            "workers": simulation["workers"],
            "schema_version": "1.1",
            "model_version": "material-generation-1.5-earth-sinr",
        },
        "export": {
            "formats": ["csv"],
            "max_records_per_file": 100000,
            "max_file_size_bytes": 268435456,
        },
        "metadata": {
            "description": "GraphLSTM受控参数覆盖场景",
            "training_group_id": f"graph-family-{family + 1:02d}",
            "parameter_version": config["parameter_version"],
            "jammer_profile": profile,
            "coordinate_frame": "LOCAL_ENU",
            "reference_data_is_calibration_only": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    expected = config["family_count"] * config["replicates_per_family"]
    if expected != config["scenario_count"]:
        raise ValueError("family_count × replicates_per_family 必须等于 scenario_count")
    random.Random(config["random_seed"])  # 明确固定计划版本的随机种子。
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for family in range(config["family_count"]):
        for replicate in range(config["replicates_per_family"]):
            scenario = build_scenario(config, family, replicate)
            path = args.output / f"{scenario['scene_id']}.json"
            path.write_text(
                json.dumps(scenario, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            group = scenario["node_groups"][0]
            index.append(
                {
                    "file": path.name,
                    "scene_id": scenario["scene_id"],
                    "training_group_id": scenario["metadata"]["training_group_id"],
                    "node_count": group["count"],
                    "formation": group["formation"],
                    "speed_mps": group["speed_mps"],
                    "spacing_m": group["spacing_m"],
                    "duration_s": scenario["simulation"]["duration_s"],
                    "jammer_profile": scenario["metadata"]["jammer_profile"],
                    "random_seed": scenario["simulation"]["random_seed"],
                }
            )
    (args.output / "index.json").write_text(
        json.dumps(
            {
                "parameter_version": config["parameter_version"],
                "scenario_count": len(index),
                "scenarios": index,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "scenario_count": len(index)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
