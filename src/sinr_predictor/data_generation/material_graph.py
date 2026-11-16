"""将算料生成 1.5 的分卷 CSV 转换为因果图训练任务。

转换后每个仿真运行是一个分片，每个时隙是一张节点和有向链路图。
原始本地球面 ENU 坐标转换为图模型外部接口要求的经纬高和 NED 向量。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

EARTH_RADIUS_M = 6_371_000.0
ORIGIN_LAT_DEG = 39.9
ORIGIN_LON_DEG = 116.4

REQUIRED_NODE_FIELDS = {
    "scene_id",
    "run_id",
    "step_index",
    "sim_time_s",
    "node_id",
    "group_id",
    "active",
    "is_missing",
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "velocity_x_mps",
    "velocity_y_mps",
    "velocity_z_mps",
    "acceleration_x_mps2",
    "acceleration_y_mps2",
    "acceleration_z_mps2",
    "heading_deg",
    "pitch_deg",
    "roll_deg",
}
REQUIRED_LINK_FIELDS = {
    "scene_id",
    "run_id",
    "step_index",
    "sim_time_s",
    "source_node_id",
    "destination_node_id",
    "connected",
    "is_missing",
    "sinr_db",
    "calculation_status",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_boolean(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{field} 必须为 true/false 或 1/0，实际为 {value!r}")


def _dot(left, right) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _add_scaled(*terms) -> tuple[float, float, float]:
    return tuple(sum(scale * vector[index] for scale, vector in terms) for index in range(3))


@dataclass(frozen=True, slots=True)
class LocalFrame:
    origin_lat_deg: float = ORIGIN_LAT_DEG
    origin_lon_deg: float = ORIGIN_LON_DEG

    def __post_init__(self) -> None:
        if not math.isfinite(self.origin_lat_deg) or not -90 <= self.origin_lat_deg <= 90:
            raise ValueError("原点纬度无效")
        if not math.isfinite(self.origin_lon_deg):
            raise ValueError("原点经度无效")

    @property
    def basis(self):
        latitude, longitude = map(
            math.radians, (self.origin_lat_deg, self.origin_lon_deg)
        )
        up = (
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
        )
        east = (-math.sin(longitude), math.cos(longitude), 0.0)
        north = (
            -math.sin(latitude) * math.cos(longitude),
            -math.sin(latitude) * math.sin(longitude),
            math.cos(latitude),
        )
        return up, east, north

    def point(self, east_m: float, north_m: float, altitude_m: float):
        """返回经纬高、当地单位向量及地表投影微分。"""
        if any(not math.isfinite(value) for value in (east_m, north_m, altitude_m)):
            raise ValueError("本地坐标必须为有限数")
        origin_up, origin_east, origin_north = self.basis
        distance = math.hypot(east_m, north_m)
        if distance == 0:
            up, dx, dy = origin_up, origin_east, origin_north
        else:
            east_ratio, north_ratio = east_m / distance, north_m / distance
            theta = distance / EARTH_RADIUS_M
            sine, cosine = math.sin(theta), math.cos(theta)
            tangent = _add_scaled(
                (east_ratio, origin_east), (north_ratio, origin_north)
            )
            up = _add_scaled((cosine, origin_up), (sine, tangent))
            radial = _add_scaled((-sine, origin_up), (cosine, tangent))
            sinc = sine / theta
            dx = tuple(
                east_ratio * radial[index]
                + sinc * (origin_east[index] - east_ratio * tangent[index])
                for index in range(3)
            )
            dy = tuple(
                north_ratio * radial[index]
                + sinc * (origin_north[index] - north_ratio * tangent[index])
                for index in range(3)
            )
        latitude = math.degrees(math.asin(max(-1.0, min(1.0, up[2]))))
        longitude = math.degrees(math.atan2(up[1], up[0]))
        return latitude, longitude, altitude_m, up, dx, dy

    def vector_to_ned(
        self,
        east_m: float,
        north_m: float,
        altitude_m: float,
        vector_east: float,
        vector_north: float,
        vector_up: float,
    ) -> tuple[float, float, float]:
        if any(
            not math.isfinite(value) for value in (vector_east, vector_north, vector_up)
        ):
            raise ValueError("本地向量必须为有限数")
        latitude, longitude, _, up, dx, dy = self.point(
            east_m, north_m, altitude_m
        )
        scale = (EARTH_RADIUS_M + altitude_m) / EARTH_RADIUS_M
        ecef = _add_scaled(
            (scale * vector_east, dx),
            (scale * vector_north, dy),
            (vector_up, up),
        )
        lat, lon = math.radians(latitude), math.radians(longitude)
        east = (-math.sin(lon), math.cos(lon), 0.0)
        north = (
            -math.sin(lat) * math.cos(lon),
            -math.sin(lat) * math.sin(lon),
            math.cos(lat),
        )
        return _dot(ecef, north), _dot(ecef, east), -_dot(ecef, up)


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_version") != "1.0"
        or manifest.get("status") != "COMPLETED"
        or not isinstance(manifest.get("run_id"), str)
        or not isinstance(manifest.get("scene_id"), str)
        or not isinstance(manifest.get("parts"), list)
    ):
        raise ValueError(f"manifest 不完整或不支持：{path}")
    return manifest


def _csv_parts(manifest_path: Path, manifest: dict, kind: str) -> list[Path]:
    root = manifest_path.parent.resolve()
    selected = []
    for part in manifest["parts"]:
        if not isinstance(part, dict) or part.get("kind") != kind:
            continue
        if part.get("format") != "csv" or not isinstance(part.get("file"), str):
            raise ValueError(f"{kind} 分卷格式无效：{manifest_path}")
        candidate = (root / part["file"]).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"{kind} 分卷路径无效：{candidate}")
        if sha256(candidate) != part.get("sha256"):
            raise ValueError(f"{kind} 分卷 SHA-256 不一致：{candidate}")
        selected.append(candidate)
    if not selected:
        raise ValueError(f"manifest 没有 {kind} CSV：{manifest_path}")
    return sorted(selected, key=lambda item: item.name)


def _rows(paths: list[Path], required: set[str]):
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if required - set(fields) or len(fields) != len(set(fields)):
                raise ValueError(f"{path} 缺少必需列或存在重复列")
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path} 第 {line} 行列数不一致")
                yield path, line, row


def _scenario(manifest_path: Path, manifest: dict) -> dict:
    path = manifest_path.parent.parent / f"{manifest['run_id']}_scenario.json"
    if not path.is_file():
        raise ValueError(f"缺少场景快照：{path}")
    scenario = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(scenario, dict) or scenario.get("scene_id") != manifest["scene_id"]:
        raise ValueError(f"场景快照与 manifest 不一致：{path}")
    return scenario


def _finite(row: dict, names: tuple[str, ...], path: Path, line: int) -> list[float]:
    try:
        values = [float(row[name]) for name in names]
    except ValueError as error:
        raise ValueError(f"{path} 第 {line} 行存在非数值字段") from error
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{path} 第 {line} 行存在非有限数")
    return values


def _node_record(row: dict, timestamp_ms: int, frame: LocalFrame, path: Path, line: int) -> dict:
    position = _finite(
        row, ("position_x_m", "position_y_m", "position_z_m"), path, line
    )
    velocity = _finite(
        row, ("velocity_x_mps", "velocity_y_mps", "velocity_z_mps"), path, line
    )
    acceleration = _finite(
        row,
        ("acceleration_x_mps2", "acceleration_y_mps2", "acceleration_z_mps2"),
        path,
        line,
    )
    heading, pitch, roll = _finite(
        row, ("heading_deg", "pitch_deg", "roll_deg"), path, line
    )
    latitude, longitude, altitude, *_ = frame.point(*position)
    velocity_ned = frame.vector_to_ned(*position, *velocity)
    acceleration_ned = frame.vector_to_ned(*position, *acceleration)
    active = parse_boolean(row["active"], "active")
    missing = parse_boolean(row["is_missing"], "is_missing")
    return {
        "node_id": row["node_id"].strip(),
        "observation_time_ms": timestamp_ms,
        "member_present": int(active and not missing),
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "altitude_m": altitude,
        "velocity_north_mps": velocity_ned[0],
        "velocity_east_mps": velocity_ned[1],
        "velocity_down_mps": velocity_ned[2],
        "accel_north_mps2": acceleration_ned[0],
        "accel_east_mps2": acceleration_ned[1],
        "accel_down_mps2": acceleration_ned[2],
        "pitch_deg": pitch,
        "yaw_deg": heading,
        "roll_deg": roll,
    }


@dataclass(frozen=True, slots=True)
class RunDescriptor:
    manifest_path: Path
    run_id: str
    scene_id: str
    training_group_id: str


def discover_runs(run_root: Path) -> list[RunDescriptor]:
    descriptors = []
    for path in sorted(run_root.resolve().glob("*/export/manifest.json")):
        manifest = _load_manifest(path)
        scenario = _scenario(path, manifest)
        metadata = scenario.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"场景 metadata 无效：{manifest['scene_id']}")
        group = metadata.get("training_group_id", manifest["scene_id"])
        if not isinstance(group, str) or not group.strip() or group != group.strip():
            raise ValueError(f"训练场景族无效：{manifest['scene_id']}")
        descriptors.append(
            RunDescriptor(path.resolve(), manifest["run_id"], manifest["scene_id"], group)
        )
    if not descriptors:
        raise ValueError(f"没有找到完整运行：{run_root}")
    if len({item.run_id for item in descriptors}) != len(descriptors):
        raise ValueError("存在重复 run_id")
    return sorted(descriptors, key=lambda item: item.scene_id)


def convert_run(descriptor: RunDescriptor, output: Path, frame: LocalFrame) -> dict:
    manifest = _load_manifest(descriptor.manifest_path)
    scenario = _scenario(descriptor.manifest_path, manifest)
    simulation = scenario.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError(f"场景缺少 simulation：{descriptor.scene_id}")
    try:
        step_s = float(simulation["step_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"场景仿真步长无效：{descriptor.scene_id}") from error
    if not math.isclose(step_s, 1.0, abs_tol=1e-9):
        raise ValueError(f"图训练当前只接受 1 秒时隙：{descriptor.scene_id}")

    node_steps: dict[int, list[dict]] = {}
    step_times: dict[int, float] = {}
    node_ids = set()
    seen_nodes = set()
    for path, line, row in _rows(
        _csv_parts(descriptor.manifest_path, manifest, "nodes"), REQUIRED_NODE_FIELDS
    ):
        if row["run_id"] != descriptor.run_id or row["scene_id"] != descriptor.scene_id:
            raise ValueError(f"{path} 第 {line} 行场景或运行编号不一致")
        try:
            step, sim_time = int(row["step_index"]), float(row["sim_time_s"])
        except ValueError as error:
            raise ValueError(f"{path} 第 {line} 行时隙无效") from error
        node_id = row["node_id"].strip()
        if step < 0 or not node_id or not math.isfinite(sim_time) or sim_time < 0:
            raise ValueError(f"{path} 第 {line} 行节点或时间无效")
        key = step, node_id
        if key in seen_nodes:
            raise ValueError(f"节点同时隙重复：{key}")
        seen_nodes.add(key)
        node_ids.add(node_id)
        previous = step_times.setdefault(step, sim_time)
        if not math.isclose(previous, sim_time, abs_tol=1e-9):
            raise ValueError(f"同时隙出现多个仿真时间：{step}")
        timestamp_ms = round(sim_time * 1000)
        node_steps.setdefault(step, []).append(
            _node_record(row, timestamp_ms, frame, path, line)
        )

    relation_steps: dict[int, list[dict]] = {step: [] for step in node_steps}
    seen_relations = set()
    valid_relations = 0
    for path, line, row in _rows(
        _csv_parts(descriptor.manifest_path, manifest, "links"), REQUIRED_LINK_FIELDS
    ):
        if row["run_id"] != descriptor.run_id or row["scene_id"] != descriptor.scene_id:
            raise ValueError(f"{path} 第 {line} 行场景或运行编号不一致")
        try:
            step, sim_time = int(row["step_index"]), float(row["sim_time_s"])
            sinr = float(row["sinr_db"])
        except ValueError as error:
            raise ValueError(f"{path} 第 {line} 行链路数值无效") from error
        source = row["source_node_id"].strip()
        destination = row["destination_node_id"].strip()
        key = step, source, destination
        if (
            key in seen_relations
            or source == destination
            or source not in node_ids
            or destination not in node_ids
            or step not in node_steps
            or not math.isfinite(sim_time)
            or not math.isfinite(sinr)
            or not math.isclose(step_times[step], sim_time, abs_tol=1e-9)
        ):
            raise ValueError(f"{path} 第 {line} 行链路端点、时间或唯一性无效")
        seen_relations.add(key)
        valid = (
            parse_boolean(row["connected"], "connected")
            and not parse_boolean(row["is_missing"], "is_missing")
            and row["calculation_status"] == "OK"
        )
        if valid:
            relation_steps[step].append(
                {
                    "source_id": source,
                    "destination_id": destination,
                    "observation_time_ms": round(sim_time * 1000),
                    "link_quality": sinr,
                }
            )
            valid_relations += 1

    steps = sorted(node_steps)
    if len(steps) < 13 or steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError(f"场景时隙不连续或不足 13 个：{descriptor.scene_id}")
    if any(
        not math.isclose(step_times[right] - step_times[left], step_s, abs_tol=1e-9)
        for left, right in zip(steps, steps[1:], strict=False)
    ):
        raise ValueError(f"场景仿真时间不按固定步长递增：{descriptor.scene_id}")
    ordered_nodes = sorted(node_ids)
    queries = [
        {"source_id": source, "destination_id": destination}
        for source in ordered_nodes
        for destination in ordered_nodes
        if source != destination
    ]
    frames = [
        {
            "received_time_ms": round(step_times[step] * 1000),
            "node_records": sorted(node_steps[step], key=lambda row: row["node_id"]),
            "relation_records": sorted(
                relation_steps[step],
                key=lambda row: (row["source_id"], row["destination_id"]),
            ),
        }
        for step in steps
    ]
    episode = {
        "episode_id": descriptor.training_group_id,
        "cluster_id": descriptor.scene_id,
        "query_relations": queries,
        "frames": frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"episodes": [episode]}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        "file": output.as_posix(),
        "sha256": sha256(output),
        "run_id": descriptor.run_id,
        "scene_id": descriptor.scene_id,
        "training_group_id": descriptor.training_group_id,
        "frame_count": len(frames),
        "node_count": len(ordered_nodes),
        "query_count": len(queries),
        "valid_relation_count": valid_relations,
    }


def _write_index(path: Path, entries: list[dict], metadata: dict) -> None:
    shards = [{"file": entry["file"], "sha256": entry["sha256"]} for entry in entries]
    path.write_text(
        json.dumps(
            {
                "schema_version": "flight_episode_shards_v1",
                "metadata": metadata,
                "shards": shards,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def convert_exports(run_root: Path, output_dir: Path, *, smoke_groups: int = 0) -> dict:
    """把多次仿真运行转换为分片图数据，并生成完整及 smoke 索引。"""

    descriptors = discover_runs(run_root)
    output_dir = output_dir.resolve()
    shard_dir = output_dir / "episodes"
    shard_dir.mkdir(parents=True, exist_ok=True)
    frame = LocalFrame()
    entries = []
    for descriptor in descriptors:
        relative = Path("episodes") / f"{descriptor.scene_id}.json"
        result = convert_run(descriptor, output_dir / relative, frame)
        result["file"] = relative.as_posix()
        entries.append(result)
    metadata = {
        "data_kind": "simulation",
        "source": "material-generation-1.5",
        "coordinate_conversion": (
            "simulator spherical-earth local ENU to latitude/longitude/altitude and NED"
        ),
        "origin_latitude_deg": frame.origin_lat_deg,
        "origin_longitude_deg": frame.origin_lon_deg,
        "episode_grouping": "episode_id=training_group_id; cluster_id=scene_id",
        "run_count": len(entries),
        "training_group_count": len({entry["training_group_id"] for entry in entries}),
        "frame_count": sum(entry["frame_count"] for entry in entries),
        "valid_relation_count": sum(entry["valid_relation_count"] for entry in entries),
        "entries": entries,
    }
    _write_index(output_dir / "index.json", entries, metadata)
    if smoke_groups:
        if type(smoke_groups) is not int or smoke_groups < 3:
            raise ValueError("smoke_groups 必须为至少 3 的整数")
        chosen = []
        seen = set()
        for entry in entries:
            group = entry["training_group_id"]
            if group not in seen:
                seen.add(group)
                chosen.append(entry)
            if len(chosen) == smoke_groups:
                break
        if len(chosen) < smoke_groups:
            raise ValueError("可用训练场景族少于 smoke_groups")
        smoke_metadata = {
            key: value for key, value in metadata.items() if key != "entries"
        }
        smoke_metadata.update(
            run_count=len(chosen),
            training_group_count=len(chosen),
            frame_count=sum(item["frame_count"] for item in chosen),
            valid_relation_count=sum(item["valid_relation_count"] for item in chosen),
            entries=chosen,
            subset="one run from each of the first training groups",
        )
        _write_index(output_dir / "smoke_index.json", chosen, smoke_metadata)
    return metadata
