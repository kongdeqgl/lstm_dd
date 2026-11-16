import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sinr_predictor import ModelConfig
from sinr_predictor.data.dataset import FlightEpisode, build_samples, load_episodes, split_episodes
from sinr_predictor.data.normalization import Normalization
from sinr_predictor.data_generation.synthetic import make_synthetic_dataset
from sinr_predictor.models.backends import PersistenceBackend
from sinr_predictor.training.evaluate import evaluate_samples


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(ModelConfig(), history_steps=3, forecast_steps=2)
        self.raw = make_synthetic_dataset(episodes=3, steps=12)

    def episode(self, row=None):
        return FlightEpisode.from_dict(self.raw["episodes"][0] if row is None else row)

    def test_split_keeps_all_clusters_of_one_flight_together(self):
        episodes = [self.episode(row) for row in self.raw["episodes"]]
        episodes.append(replace(episodes[0], cluster_id="another-cluster"))
        splits = split_episodes(episodes)
        groups = [{episode.episode_id for episode in split} for split in splits]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        locations = [
            index
            for index, split in enumerate(splits)
            for episode in split
            if episode.episode_id == episodes[0].episode_id
        ]
        self.assertEqual(len(locations), 2)
        self.assertEqual(locations[0], locations[1])
        with self.assertRaises(ValueError):
            split_episodes(episodes[:2])

    def test_future_target_changes_do_not_change_history_input(self):
        raw = self.raw["episodes"][0]
        reference = raw["frames"][4]["received_time_ms"]
        raw["reference_times_ms"] = [reference]
        original = build_samples([self.episode()], self.config)[0]
        changed = copy.deepcopy(raw)
        changed["frames"][5]["relation_records"][0]["link_quality"] += 500.0
        actual = build_samples([self.episode(changed)], self.config)[0]
        self.assertEqual(original.windows, actual.windows)
        self.assertNotEqual(original.targets, actual.targets)

    def test_late_measurement_absent_from_historical_input(self):
        raw = self.raw["episodes"][0]
        reference = raw["frames"][4]["received_time_ms"]
        raw["reference_times_ms"] = [reference]
        late = raw["frames"][3]["relation_records"].pop(0)
        raw["frames"][5]["relation_records"].append(late)
        sample = build_samples([self.episode()], self.config)[0]
        query = next(window for window in sample.windows if window.query.key == ("N0", "N1"))
        self.assertFalse(query.observed_mask[1])
        self.assertTrue(query.observed_mask[0])

    def test_automatic_query_set_does_not_peek_at_future_edges(self):
        raw = self.raw["episodes"][0]
        raw.pop("query_relations")
        raw["reference_times_ms"] = [raw["frames"][4]["received_time_ms"]]
        for frame in raw["frames"][:5]:
            frame["relation_records"] = [
                record
                for record in frame["relation_records"]
                if (record["source_id"], record["destination_id"]) != ("N0", "N1")
            ]
        sample = build_samples([self.episode()], self.config)[0]
        self.assertNotIn(("N0", "N1"), [window.query.key for window in sample.windows])

    def test_missing_future_label_is_not_forward_filled(self):
        raw = self.raw["episodes"][0]
        raw["reference_times_ms"] = [raw["frames"][4]["received_time_ms"]]
        raw["frames"][5]["relation_records"].pop(0)
        sample = build_samples([self.episode()], self.config)[0]
        self.assertFalse(sample.target_mask[0][0])
        self.assertEqual(sample.targets[0][0], 0.0)
        self.assertTrue(sample.target_mask[0][1])

    def test_exit_and_rejoin_in_future_invalidates_old_session_labels(self):
        raw = self.raw["episodes"][0]
        raw["reference_times_ms"] = [raw["frames"][4]["received_time_ms"]]
        raw["frames"][5]["node_records"][1]["member_present"] = 0
        sample = build_samples([self.episode()], self.config)[0]
        index = next(
            i for i, window in enumerate(sample.windows) if window.query.key == ("N0", "N1")
        )
        self.assertEqual(sample.target_mask[index], (False, False))
        other = next(
            i for i, window in enumerate(sample.windows) if window.query.key == ("N0", "N2")
        )
        self.assertEqual(sample.target_mask[other], (True, True))

    def test_scaler_ignores_missing_fields_and_deduplicates_labels(self):
        samples = build_samples([self.episode()], self.config)
        scaler = Normalization.fit(samples, self.config)
        self.assertEqual(scaler.nodes.count[6:], (0,) * 9)
        self.assertEqual(scaler.nodes.mean[6:], (0.0,) * 9)
        self.assertEqual(scaler.nodes.scale[6:], (1.0,) * 9)
        labels = {}
        for sample in samples:
            for index in sample.eligible_indices:
                for step, (target, valid) in enumerate(
                    zip(sample.targets[index], sample.target_mask[index], strict=True), start=1
                ):
                    if valid:
                        labels[
                            (
                                *sample.windows[index].query.key,
                                sample.reference_time_ms + step * 1000,
                            )
                        ] = target
        self.assertEqual(scaler.target.count[0], len(labels))
        self.assertAlmostEqual(scaler.target.mean[0], sum(labels.values()) / len(labels))
        self.assertEqual(Normalization.from_dict(scaler.to_dict()), scaler)

    def test_no_labels_is_reported_as_no_data(self):
        samples = build_samples([self.episode()], self.config)
        sample = replace(samples[0], target_mask=tuple((False, False) for _ in samples[0].windows))
        result = evaluate_samples(PersistenceBackend(), [sample], self.config)
        self.assertIsNone(result["overall_rmse_db"])
        self.assertEqual(result["target_missing_ratio"], 1.0)
        self.assertTrue(all(row["mae_db"] is None for row in result["horizons"]))

    def test_reject_impossible_arrival_time(self):
        self.raw["episodes"][0]["frames"][0]["received_time_ms"] -= 1
        with self.assertRaises(ValueError):
            self.episode()

    def test_load_sharded_episode_index_and_verify_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "episode.json"
            shard.write_text(
                json.dumps({"episodes": [self.raw["episodes"][0]]}), encoding="utf-8"
            )
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()
            index = root / "index.json"
            index.write_text(
                json.dumps(
                    {
                        "schema_version": "flight_episode_shards_v1",
                        "metadata": {"purpose": "test"},
                        "shards": [{"file": "episode.json", "sha256": digest}],
                    }
                ),
                encoding="utf-8",
            )
            episodes = load_episodes(root)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].cluster_id, "cluster-0")
            shard.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_episodes(index)


if __name__ == "__main__":
    unittest.main()
