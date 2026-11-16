import csv
import json
import tempfile
import unittest
from pathlib import Path

from sinr_predictor import ModelConfig
from sinr_predictor.data.dataset import build_samples, load_episodes
from sinr_predictor.data_generation.material_graph import LocalFrame, convert_exports, sha256


class MaterialGraphTests(unittest.TestCase):
    def test_local_frame_origin_and_vector_convention(self):
        frame = LocalFrame()
        latitude, longitude, altitude, *_ = frame.point(0.0, 0.0, 12_000.0)
        self.assertAlmostEqual(latitude, frame.origin_lat_deg)
        self.assertAlmostEqual(longitude, frame.origin_lon_deg)
        self.assertEqual(altitude, 12_000.0)
        north, east, down = frame.vector_to_ned(0, 0, 12_000, 10, 20, 3)
        horizontal_scale = (6_371_000.0 + 12_000.0) / 6_371_000.0
        self.assertAlmostEqual(north, 20.0 * horizontal_scale, places=6)
        self.assertAlmostEqual(east, 10.0 * horizontal_scale, places=6)
        self.assertAlmostEqual(down, -3.0, places=6)

    @staticmethod
    def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_convert_exports_builds_causal_graph_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run-1"
            export = run / "export"
            export.mkdir(parents=True)
            node_fields = [
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
            ]
            link_fields = [
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
            ]
            nodes, links = [], []
            for step in range(13):
                for index, node_id in enumerate(("N1", "N2")):
                    nodes.append(
                        {
                            "scene_id": "scene-1",
                            "run_id": "run-1",
                            "step_index": step,
                            "sim_time_s": float(step),
                            "node_id": node_id,
                            "group_id": "G1",
                            "active": "True",
                            "is_missing": "False",
                            "position_x_m": index * 1000 + step * 10,
                            "position_y_m": index * 100,
                            "position_z_m": 20_000,
                            "velocity_x_mps": 10,
                            "velocity_y_mps": 0,
                            "velocity_z_mps": 0,
                            "acceleration_x_mps2": 0,
                            "acceleration_y_mps2": 0,
                            "acceleration_z_mps2": 0,
                            "heading_deg": 0,
                            "pitch_deg": 0,
                            "roll_deg": 0,
                        }
                    )
                for source, destination, offset in (("N1", "N2", 0), ("N2", "N1", 1)):
                    links.append(
                        {
                            "scene_id": "scene-1",
                            "run_id": "run-1",
                            "step_index": step,
                            "sim_time_s": float(step),
                            "source_node_id": source,
                            "destination_node_id": destination,
                            "connected": "True",
                            "is_missing": "False",
                            "sinr_db": 20.0 + step + offset,
                            "calculation_status": "OK",
                        }
                    )
            node_path, link_path = export / "nodes.csv", export / "links.csv"
            self.write_csv(node_path, node_fields, nodes)
            self.write_csv(link_path, link_fields, links)
            manifest = {
                "manifest_version": "1.0",
                "run_id": "run-1",
                "scene_id": "scene-1",
                "status": "COMPLETED",
                "parts": [
                    {
                        "file": node_path.name,
                        "kind": "nodes",
                        "format": "csv",
                        "sha256": sha256(node_path),
                    },
                    {
                        "file": link_path.name,
                        "kind": "links",
                        "format": "csv",
                        "sha256": sha256(link_path),
                    },
                ],
            }
            (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run / "run-1_scenario.json").write_text(
                json.dumps(
                    {
                        "scene_id": "scene-1",
                        "simulation": {"step_s": 1.0},
                        "metadata": {"training_group_id": "family-1"},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "graph"
            metadata = convert_exports(root, output)
            self.assertEqual(metadata["frame_count"], 13)
            self.assertEqual(metadata["valid_relation_count"], 26)
            episodes = load_episodes(output)
            self.assertEqual((episodes[0].episode_id, episodes[0].cluster_id), ("family-1", "scene-1"))
            self.assertEqual(len(episodes[0].frames), 13)
            config = ModelConfig(history_steps=10, forecast_steps=3)
            samples = build_samples(episodes, config)
            self.assertEqual(len(samples), 1)
            self.assertEqual(len(samples[0].eligible_indices), 2)


if __name__ == "__main__":
    unittest.main()
