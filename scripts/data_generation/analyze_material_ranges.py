"""流式汇总算料导出的关键数值范围，作为训练场景参数参考。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


class Values:
    def __init__(self) -> None:
        self.items: list[float] = []

    def add(self, value) -> None:
        number = float(value)
        if math.isfinite(number):
            self.items.append(number)

    def summary(self) -> dict[str, float | int | None]:
        if not self.items:
            return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
        values = sorted(self.items)

        def percentile(ratio: float) -> float:
            position = ratio * (len(values) - 1)
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            weight = position - lower
            return values[lower] * (1 - weight) + values[upper] * weight

        return {
            "count": len(values),
            "min": values[0],
            "p05": percentile(0.05),
            "median": percentile(0.5),
            "p95": percentile(0.95),
            "max": values[-1],
        }


def csv_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics: dict[str, Values] = defaultdict(Values)
    scenario_counts: dict[str, int] = defaultdict(int)
    run_count = 0

    for manifest_path in sorted(args.run_root.glob("*/export/manifest.json")):
        run_count += 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        run_dir = manifest_path.parent.parent
        scenario_path = run_dir / f"{manifest['run_id']}_scenario.json"
        scenario = json.loads(scenario_path.read_text(encoding="utf-8-sig"))
        simulation = scenario["simulation"]
        metrics["simulation.step_s"].add(simulation["step_s"])
        metrics["simulation.duration_s"].add(simulation["duration_s"])
        expanded_nodes = scenario.get("nodes", [])
        if expanded_nodes:
            metrics["scenario.node_count"].add(len(expanded_nodes))
        for group in scenario.get("node_groups", []):
            if "count" in group:
                metrics["scenario.node_count"].add(group["count"])
            if "spacing_m" in group:
                metrics["scenario.spacing_m"].add(group["spacing_m"])
            if "speed_mps" in group:
                metrics["scenario.speed_mps"].add(group["speed_mps"])
            if "formation" in group:
                scenario_counts[f"formation:{group['formation']}"] += 1
        for jammer in scenario.get("jammers", []):
            metrics["scenario.jammer_strength_w"].add(jammer["strength_w"])
            metrics["scenario.jammer_effect_start_s"].add(jammer["effect_start_s"])
            metrics["scenario.jammer_effect_end_s"].add(jammer["effect_end_s"])
            scenario_counts[f"jammer_trajectory:{jammer['trajectory_type']}"] += 1

        parts = manifest.get("parts", [])
        for part in parts:
            if part.get("format") != "csv":
                continue
            path = manifest_path.parent / part["file"]
            if part.get("kind") == "nodes":
                for row in csv_rows(path):
                    velocity = math.hypot(
                        float(row["velocity_x_mps"]),
                        float(row["velocity_y_mps"]),
                        float(row["velocity_z_mps"]),
                    )
                    acceleration = math.hypot(
                        float(row["acceleration_x_mps2"]),
                        float(row["acceleration_y_mps2"]),
                        float(row["acceleration_z_mps2"]),
                    )
                    metrics["node.altitude_m"].add(row["position_z_m"])
                    metrics["node.speed_mps"].add(velocity)
                    metrics["node.acceleration_mps2"].add(acceleration)
                    metrics["node.pitch_deg"].add(row["pitch_deg"])
                    metrics["node.roll_deg"].add(row["roll_deg"])
            elif part.get("kind") == "links":
                for row in csv_rows(path):
                    for field in (
                        "distance_m",
                        "path_loss_db",
                        "receive_power_dbm",
                        "interference_power_w",
                        "sinr_db",
                    ):
                        metrics[f"link.{field}"].add(row[field])

    output = {
        "run_count": run_count,
        "reference_only": True,
        "metrics": {name: values.summary() for name, values in sorted(metrics.items())},
        "categories": dict(sorted(scenario_counts.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
