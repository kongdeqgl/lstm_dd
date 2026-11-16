import copy
import json
import math
import random
import unittest
from dataclasses import replace

from sinr_predictor import ModelConfig, Predictor, RequestStatus, ResultStatus
from sinr_predictor.data.history import HistoryCache
from sinr_predictor.data.preprocess import (
    NODE_FEATURE_NAMES,
    build_query_window,
    build_snapshots,
    geodetic_to_ecef,
    ned_to_ecef,
)
from sinr_predictor.demo import make_demo_request
from sinr_predictor.schemas import PredictionRequest, QueryRelation


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.now = 10_000
        self.config = replace(
            ModelConfig(), history_steps=4, min_observation_ratio=0.75,
            max_consecutive_missing_steps=1,
        )
        self.payload = make_demo_request(self.now, self.config)

    def predictor(self, **kwargs):
        return Predictor(self.config, clock_ms=lambda: self.now, **kwargs)

    def results(self, payload=None, **kwargs):
        response = self.predictor(**kwargs).predict(self.payload if payload is None else payload)
        self.assertEqual(response.status_code, RequestStatus.SUCCESS, response.diagnostic)
        return response.results

    def test_end_to_end_statuses_targets_and_values(self):
        response = self.predictor().predict(self.payload)
        self.assertEqual(response.status_code, RequestStatus.SUCCESS)
        self.assertEqual(len(response.results), 12)
        for index, expected in enumerate((0, 3, 2, 4)):
            group = response.results[index * 3:index * 3 + 3]
            self.assertEqual([row.status_code for row in group], [expected] * 3)
            self.assertEqual([row.target_time_ms for row in group], [11_000, 12_000, 13_000])
            self.assertEqual([row.predicted_value for row in group], [11.5 if expected == 0 else None] * 3)
        encoded = response.to_dict()
        self.assertNotIn("diagnostic", encoded)
        self.assertEqual(json.loads(json.dumps(encoded, allow_nan=False)), encoded)

    def test_opposite_direction_uses_its_own_history(self):
        self.payload["relation_records"] += [
            dict(row, source_id="B", destination_id="A", link_quality=-8.0)
            for row in self.payload["relation_records"]
            if row["source_id"] == "A" and row["observation_time_ms"] != 7000
        ]
        results = self.results()
        self.assertEqual([row.predicted_value for row in results[3:6]], [-8.0] * 3)

    def test_zero_is_a_valid_measurement(self):
        self.payload["relation_records"][-2]["link_quality"] = 0.0
        result = self.results()[0]
        self.assertEqual(result.status_code, ResultStatus.SUCCESS)
        self.assertEqual(result.predicted_value, 0.0)

    def test_short_hold_does_not_manufacture_observation_coverage(self):
        config = replace(self.config, max_link_age_ms=20_000)
        self.payload["relation_records"] = self.payload["relation_records"][:1]
        predictor = Predictor(config, clock_ms=lambda: self.now)
        response = predictor.predict(self.payload)
        self.assertEqual(response.results[0].status_code, ResultStatus.INSUFFICIENT_HISTORY)

    def test_contiguous_gap_checked_separately_from_coverage(self):
        config = replace(self.config, min_observation_ratio=0.5)
        self.payload["relation_records"] = [
            row for row in self.payload["relation_records"]
            if row["source_id"] == "A" and row["observation_time_ms"] in (7000, 10_000)
        ]
        response = Predictor(config, clock_ms=lambda: self.now).predict(self.payload)
        self.assertEqual(response.results[0].status_code, ResultStatus.INSUFFICIENT_HISTORY)

    def test_missing_numeric_fields_are_not_observed_zeros(self):
        last_a = next(
            row for row in self.payload["node_records"]
            if row["node_id"] == "A" and row["observation_time_ms"] == self.now
        )
        del last_a["velocity_north_mps"]
        self.assertEqual(self.results()[0].status_code, ResultStatus.UNAVAILABLE)

    def test_missing_node_member_and_self_queries(self):
        self.payload["query_relations"] = [{"source_id": "A", "destination_id": "A"}]
        self.assertEqual(self.results()[0].status_code, ResultStatus.UNAVAILABLE)
        last_b = next(
            row for row in self.payload["node_records"]
            if row["node_id"] == "B" and row["observation_time_ms"] == self.now
        )
        last_b["member_present"] = 0
        self.payload["query_relations"] = [{"source_id": "A", "destination_id": "B"}]
        self.assertEqual(self.results()[0].status_code, ResultStatus.UNAVAILABLE)

    def test_stale_endpoint(self):
        self.payload["node_records"] = [
            row for row in self.payload["node_records"]
            if row["node_id"] != "B" or row["observation_time_ms"] == 7000
        ]
        self.assertEqual(self.results()[0].status_code, ResultStatus.STALE_OBSERVATION)

    def test_member_rejoin_does_not_resume_old_link_history(self):
        leaving = dict(next(row for row in self.payload["node_records"] if row["node_id"] == "B"))
        leaving.update(observation_time_ms=9500, member_present=0)
        self.payload["node_records"].append(leaving)
        self.assertEqual(self.results()[0].status_code, ResultStatus.INSUFFICIENT_HISTORY)

    def test_long_interruption_starts_new_session(self):
        config = replace(self.config, continuity_gap_ms=2000)
        self.payload["node_records"] = [
            row for row in self.payload["node_records"]
            if row["node_id"] != "B" or row["observation_time_ms"] in (7000, 10_000)
        ]
        response = Predictor(config, clock_ms=lambda: self.now).predict(self.payload)
        self.assertEqual(response.results[0].status_code, ResultStatus.INSUFFICIENT_HISTORY)

    def test_invalid_request_never_partially_updates_cache(self):
        predictor = self.predictor()
        future = dict(self.payload["node_records"][0], observation_time_ms=self.now + 1)
        self.payload["node_records"].append(future)
        response = predictor.predict(self.payload)
        self.assertEqual(response.status_code, RequestStatus.INVALID_REQUEST)
        self.assertEqual(predictor.history.nodes, {})

    def test_conflicting_duplicate_batch_is_atomic(self):
        predictor = self.predictor()
        predictor.predict(self.payload)
        extra = dict(self.payload["node_records"][0], node_id="NEW")
        conflict = dict(self.payload["node_records"][0], altitude_m=500.0)
        request = dict(self.payload, node_records=[extra, conflict], relation_records=[])
        response = predictor.predict(request)
        self.assertEqual(response.status_code, RequestStatus.INVALID_REQUEST)
        self.assertNotIn("NEW", predictor.history.nodes)

    def test_identical_duplicates_are_idempotent(self):
        predictor = self.predictor()
        first = predictor.predict(self.payload)
        self.payload["node_records"] *= 2
        self.payload["relation_records"] *= 2
        second = predictor.predict(self.payload)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(len(predictor.history.nodes["A"]), 4)

    def test_record_and_query_permutations_preserve_link_results(self):
        original = self.predictor().predict(self.payload)
        reordered = copy.deepcopy(self.payload)
        random.Random(42).shuffle(reordered["node_records"])
        random.Random(24).shuffle(reordered["relation_records"])
        reordered["query_relations"].reverse()
        actual = self.predictor().predict(reordered)
        def keyed(response):
            return {
                (row.source_id, row.destination_id, row.target_time_ms):
                (row.status_code, row.predicted_value)
                for row in response.results
            }
        self.assertEqual(keyed(original), keyed(actual))

    def test_duplicate_queries_keep_original_positions(self):
        self.payload["query_relations"] = [self.payload["query_relations"][0]] * 2
        results = self.results()
        self.assertEqual(results[:3], results[3:])

    def test_incremental_request_reuses_cached_history(self):
        predictor = self.predictor()
        predictor.predict(self.payload)
        self.now = 11_000
        nodes = [dict(row, observation_time_ms=self.now) for row in self.payload["node_records"][-3:]]
        next_request = {
            "request_id": "incremental",
            "node_records": nodes,
            "relation_records": [{
                "source_id": "A", "destination_id": "B",
                "observation_time_ms": self.now, "link_quality": 20.0,
            }],
            "query_relations": self.payload["query_relations"][:1],
        }
        response = predictor.predict(next_request)
        self.assertEqual(response.results[0].status_code, ResultStatus.SUCCESS)
        self.assertEqual(response.results[0].predicted_value, 20.0)

    def test_cluster_head_receive_time_fills_missing_observation_time(self):
        nodes = [
            {key: value for key, value in row.items() if key != "observation_time_ms"}
            for row in self.payload["node_records"]
            if row["observation_time_ms"] == self.now
        ]
        relation = next(
            row
            for row in self.payload["relation_records"]
            if row["observation_time_ms"] == self.now
        )
        relation = {
            key: value for key, value in relation.items() if key != "observation_time_ms"
        }
        payload = {
            "request_id": "cluster-head-time",
            "node_records": nodes,
            "relation_records": [relation],
            "query_relations": [],
        }
        # 离线数据解析仍要求明确时间，只有 Predictor 在线入口自动打时间戳。
        with self.assertRaises(ValueError):
            PredictionRequest.from_dict(payload)
        predictor = self.predictor()
        response = predictor.predict(payload)
        self.assertEqual(response.status_code, RequestStatus.SUCCESS, response.diagnostic)
        self.assertTrue(nodes)
        for row in nodes:
            self.assertIn(self.now, predictor.history.nodes[row["node_id"]])
        key = relation["source_id"], relation["destination_id"]
        self.assertIn(self.now, predictor.history.relations[key])

    def test_explicit_source_time_is_preserved_when_receive_time_default_is_enabled(self):
        row = dict(self.payload["node_records"][0])
        payload = {
            "request_id": "source-time",
            "node_records": [row],
            "relation_records": [],
            "query_relations": [],
        }
        predictor = self.predictor()
        response = predictor.predict(payload)
        self.assertEqual(response.status_code, RequestStatus.SUCCESS, response.diagnostic)
        self.assertIn(row["observation_time_ms"], predictor.history.nodes[row["node_id"]])

    def test_expired_target_is_not_returned_as_success(self):
        clock = iter((10_000, 11_000))
        response = Predictor(self.config, clock_ms=lambda: next(clock)).predict(self.payload)
        self.assertEqual(response.results[0].status_code, ResultStatus.UNAVAILABLE)
        self.assertIsNone(response.results[0].predicted_value)
        self.assertEqual(response.results[1].status_code, ResultStatus.SUCCESS)

    def test_unavailable_or_failed_backend_has_request_level_status(self):
        self.assertEqual(
            self.predictor(backend=None).predict(self.payload).status_code,
            RequestStatus.MODEL_UNAVAILABLE,
        )
        class BadBackend:
            def predict(self, windows, config):
                return [[math.nan] * config.forecast_steps] * len(windows)
        response = self.predictor(backend=BadBackend()).predict(self.payload)
        self.assertEqual(response.status_code, RequestStatus.MODEL_UNAVAILABLE)
        self.assertEqual(response.results, ())

    def test_ineligible_queries_bypass_backend(self):
        class NeverCalled:
            def predict(self, windows, config):
                raise AssertionError("无效链路不能进入数值计算")
        self.payload["query_relations"] = self.payload["query_relations"][2:]
        response = self.predictor(backend=NeverCalled()).predict(self.payload)
        self.assertEqual(response.status_code, RequestStatus.SUCCESS)
        self.assertEqual([row.status_code for row in response.results[::3]], [2, 4])

    def test_empty_query(self):
        payload = dict(self.payload, query_relations=[])
        response = self.predictor().predict(payload)
        self.assertEqual(response.status_code, RequestStatus.SUCCESS)
        self.assertEqual(response.results, ())

    def test_invalid_types_numbers_and_unknown_fields(self):
        for invalid in (True, "10", math.nan, math.inf, 10**400):
            with self.subTest(invalid=repr(invalid)):
                payload = copy.deepcopy(self.payload)
                payload["relation_records"][0]["link_quality"] = invalid
                self.assertEqual(
                    self.predictor().predict(payload).status_code, RequestStatus.INVALID_REQUEST,
                )
        payload = dict(self.payload, reference_time_ms=self.now)
        self.assertEqual(self.predictor().predict(payload).status_code, RequestStatus.INVALID_REQUEST)
        payload = copy.deepcopy(self.payload)
        payload["node_records"][0]["observation_time_ms"] = True
        self.assertEqual(self.predictor().predict(payload).status_code, RequestStatus.INVALID_REQUEST)

    def test_clock_rollback_rejected_without_rewriting_history(self):
        predictor = self.predictor()
        predictor.predict(self.payload)
        self.now = 9000
        response = predictor.predict(self.payload)
        self.assertEqual(response.status_code, RequestStatus.INVALID_REQUEST)
        self.assertIn("时间必须非递减", response.diagnostic)


class CausalityAndFeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(ModelConfig(), history_steps=4)
        self.payload = make_demo_request(10_000, self.config)

    def test_late_arrival_cannot_change_prior_cutoff_snapshot(self):
        self.payload["relation_records"] = [
            row for row in self.payload["relation_records"]
            if row["observation_time_ms"] != 9000
        ]
        history = HistoryCache()
        history.ingest(PredictionRequest.from_dict(self.payload), 10_000)
        late = dict(
            self.payload, node_records=[], relation_records=[{
                "source_id": "A", "destination_id": "B",
                "observation_time_ms": 9000, "link_quality": 99.0,
            }],
        )
        history.ingest(PredictionRequest.from_dict(late), 11_000)
        self.assertEqual(history.latest_relation(("A", "B"), 9500, 10_000).link_quality, 10.5)
        self.assertEqual(history.latest_relation(("A", "B"), 9500, 11_000).link_quality, 99.0)

    def test_grid_point_never_uses_later_event(self):
        history = HistoryCache()
        history.ingest(PredictionRequest.from_dict(self.payload), 10_000)
        snapshots = build_snapshots(
            history, (QueryRelation("A", "B"),), 10_000, 10_000, self.config,
        )
        self.assertEqual([graph.time_ms for graph in snapshots], [7000, 8000, 9000, 10_000])
        self.assertEqual([graph.edges[("A", "B")].link_quality for graph in snapshots], [10, 10.5, 11, 11.5])
        self.assertEqual([graph.edges[("A", "B")].quality_age_ms for graph in snapshots], [0] * 4)

    def test_optional_feature_masks_and_directed_geometry(self):
        history = HistoryCache()
        history.ingest(PredictionRequest.from_dict(self.payload), 10_000)
        graph = build_snapshots(history, (), 10_000, 10_000, self.config)[-1]
        node = graph.nodes["A"]
        self.assertEqual(len(node.values), len(NODE_FEATURE_NAMES))
        self.assertTrue(all(node.observed_mask[:6]))
        self.assertFalse(any(node.observed_mask[6:]))
        forward, reverse = graph.edges[("A", "B")], graph.edges[("B", "A")]
        self.assertAlmostEqual(forward.values[6], reverse.values[6])
        for value, opposite in zip(forward.values[:6], reverse.values[:6], strict=True):
            self.assertAlmostEqual(value, -opposite)

    def test_hold_keeps_age_and_new_observation_flag_separate(self):
        self.payload["relation_records"] = self.payload["relation_records"][:1]
        history = HistoryCache()
        history.ingest(PredictionRequest.from_dict(self.payload), 10_000)
        config = replace(self.config, max_link_age_ms=4000)
        graphs = build_snapshots(history, (), 10_000, 10_000, config)
        window = build_query_window(
            history, graphs, QueryRelation("A", "B"), 10_000, 10_000, config,
        )
        self.assertEqual([entry.quality_age_ms for entry in window.entries], [0, 1000, 2000, 3000])
        self.assertEqual(window.observed_mask, (True, False, False, False))
        self.assertEqual(window.observation_ratio, 0.25)

    def test_cache_retention_reclaims_abandoned_identities(self):
        history = HistoryCache()
        history.ingest(PredictionRequest.from_dict(self.payload), 10_000)
        history.prune(1_000_000, 600_000)
        self.assertEqual(history.nodes, {})
        self.assertEqual(history.relations, {})

    def test_wgs84_and_ned_known_axes(self):
        self.assertEqual(geodetic_to_ecef(0, 0, 0), (6_378_137.0, 0.0, 0.0))
        self.assertEqual(ned_to_ecef(1, 2, 3, 0, 0), (-3.0, 2.0, 1.0))


class ConfigTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        for changes in (
            {"history_steps": 0}, {"forecast_steps": True},
            {"min_observation_ratio": 0}, {"min_observation_ratio": math.nan},
            {"min_observation_ratio": 10**400},
            {"cache_retention_ms": 100}, {"sampling_interval_ms": 2**32},
            {"target_metric_code": 65536}, {"model_version": "x" * 33},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(ModelConfig(), **changes)


if __name__ == "__main__":
    unittest.main()
