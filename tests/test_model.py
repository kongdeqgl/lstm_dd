import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from sinr_predictor import ModelConfig, Predictor
from sinr_predictor.data.dataset import FlightEpisode, build_samples
from sinr_predictor.data.normalization import Normalization
from sinr_predictor.data_generation.synthetic import make_synthetic_dataset

if torch is not None:
    from sinr_predictor.data.tensors import TemporalGraphBatch, TensorGraph, tensorize_windows
    from sinr_predictor.models.graph_lstm import GraphLSTM, NetworkConfig, masked_mse
    from sinr_predictor.models.neural_backend import TorchBackend, save_bundle
    from sinr_predictor.training.graph_train import GraphTrainingJob, TrainingConfig, train_model


@unittest.skipIf(torch is None, "神经网络测试需要可选算法依赖 torch")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = replace(
            ModelConfig(), model_version="test-graph-lstm", history_steps=3, forecast_steps=2
        )
        cls.network = NetworkConfig(hidden_dim=8, link_dim=8, lstm_hidden_dim=8, graph_layers=1)
        cls.raw = make_synthetic_dataset(episodes=3, steps=12)
        cls.episodes = [FlightEpisode.from_dict(row) for row in cls.raw["episodes"]]
        cls.splits = [build_samples([episode], cls.config) for episode in cls.episodes]
        cls.normalization = Normalization.fit(cls.splits[0], cls.config)
        cls.temp_root = Path(__file__).resolve().parents[1] / "outputs" / "test_tmp"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        torch.manual_seed(42)
        self.sample = self.splits[0][0]
        self.batch = tensorize_windows(
            self.sample.eligible_windows, self.normalization, self.config
        )

    def test_graph_and_temporal_parameters_receive_gradients(self):
        model = GraphLSTM(2, self.network)
        prediction = model(self.batch)
        self.assertEqual(tuple(prediction.shape), (len(self.sample.eligible_indices), 2))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        for parameter in (
            model.message_layers[0].message[0].weight,
            model.temporal.weight_ih_l0,
            model.output.weight,
        ):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_masked_nan_label_cannot_pollute_loss_or_gradient(self):
        prediction = torch.tensor([[3.0, 2.0]], requires_grad=True)
        target = torch.tensor([[float("nan"), 1.0]])
        mask = torch.tensor([[False, True]])
        loss = masked_mse(prediction, target, mask)
        self.assertEqual(float(loss.detach()), 1.0)
        loss.backward()
        self.assertEqual(prediction.grad.tolist(), [[0.0, 2.0]])
        self.assertIsNone(masked_mse(prediction, target, torch.zeros_like(mask)))

    def test_physical_node_edge_and_query_permutation_equivariance(self):
        model = GraphLSTM(2, self.network).eval()
        graphs = []
        for graph in self.batch.graphs:
            nodes = torch.arange(graph.nodes.shape[0] - 1, -1, -1)
            edges = torch.arange(graph.edges.shape[0] - 1, -1, -1)
            inverse_nodes, inverse_edges = torch.empty_like(nodes), torch.empty_like(edges)
            inverse_nodes[nodes] = torch.arange(len(nodes))
            inverse_edges[edges] = torch.arange(len(edges))
            queries = torch.arange(self.batch.num_queries - 1, -1, -1)

            def remap(values, inverse):
                output = values[queries].clone()
                valid = output >= 0
                output[valid] = inverse[output[valid]]
                return output

            graphs.append(
                TensorGraph(
                    graph.nodes[nodes],
                    graph.edges[edges],
                    inverse_nodes[graph.edge_index[:, edges]],
                    remap(graph.query_sources, inverse_nodes),
                    remap(graph.query_destinations, inverse_nodes),
                    remap(graph.query_edges, inverse_edges),
                    graph.query_valid[queries],
                )
            )
        with torch.inference_mode():
            expected = model(self.batch).flip(0)
            actual = model(TemporalGraphBatch(tuple(graphs), self.batch.num_queries))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_state_is_reset_between_windows(self):
        model = GraphLSTM(2, self.network).eval()
        with torch.inference_mode():
            first, second = model(self.batch), model(self.batch)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_graph_context_reaches_neighbor_outside_query_endpoints(self):
        window = next(
            window for window in self.sample.eligible_windows if window.query.key == ("N0", "N1")
        )
        for mode in ("graph", "endpoints"):
            with self.subTest(mode=mode):
                batch = tensorize_windows((window,), self.normalization, self.config)
                for graph in batch.graphs:
                    graph.nodes.requires_grad_(True)
                model = GraphLSTM(2, replace(self.network, spatial_mode=mode))
                model(batch).sum().backward()
                gradient = sum(float(graph.nodes.grad[2].abs().sum()) for graph in batch.graphs)
                if mode == "graph":
                    self.assertGreater(gradient, 0)
                else:
                    self.assertEqual(gradient, 0)

    def test_quality_ablation_does_not_read_motion_or_geometry(self):
        model = GraphLSTM(2, replace(self.network, spatial_mode="quality")).eval()
        graphs = []
        for graph in self.batch.graphs:
            edges = graph.edges.clone()
            edges[:, :7] += 1000
            graphs.append(replace(graph, nodes=graph.nodes + 1000, edges=edges))
        with torch.inference_mode():
            original = model(self.batch)
            changed = model(TemporalGraphBatch(tuple(graphs), self.batch.num_queries))
        torch.testing.assert_close(original, changed, atol=0, rtol=0)

    def test_changing_node_count_inside_window(self):
        raw = copy.deepcopy(self.raw["episodes"][0])
        raw["reference_times_ms"] = [raw["frames"][6]["received_time_ms"]]
        for frame in raw["frames"][:5]:
            frame["node_records"] = [row for row in frame["node_records"] if row["node_id"] != "N2"]
            frame["relation_records"] = [
                row
                for row in frame["relation_records"]
                if "N2" not in (row["source_id"], row["destination_id"])
            ]
        sample = build_samples([FlightEpisode.from_dict(raw)], self.config)[0]
        batch = tensorize_windows(sample.eligible_windows, self.normalization, self.config)
        self.assertEqual([graph.nodes.shape[0] for graph in batch.graphs], [2, 3, 3])
        self.assertTrue(torch.isfinite(GraphLSTM(2, self.network)(batch)).all())

    def test_checkpoint_roundtrip_and_online_adapter(self):
        model = GraphLSTM(2, self.network)
        backend = TorchBackend(model, self.normalization, self.config)
        expected = backend.predict(self.sample.eligible_windows, self.config)
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(self.temp_root.resolve()))
            checkpoint = Path(directory) / "model.pt"
            save_bundle(checkpoint, model, self.normalization, self.config)
            loaded = TorchBackend.load(checkpoint)
            torch.testing.assert_close(
                torch.tensor(loaded.predict(self.sample.eligible_windows, self.config)),
                torch.tensor(expected),
            )
            reference = self.sample.reference_time_ms
            frames = [
                frame
                for frame in self.raw["episodes"][0]["frames"]
                if frame["received_time_ms"] <= reference
            ]
            payload = {
                "request_id": "neural-online-test",
                "node_records": [row for frame in frames for row in frame["node_records"]],
                "relation_records": [row for frame in frames for row in frame["relation_records"]],
                "query_relations": self.raw["episodes"][0]["query_relations"],
            }
            response = Predictor.from_model_bundle(checkpoint, clock_ms=lambda: reference).predict(
                payload
            )
            self.assertTrue(
                all(row.status_code == 0 for row in response.results), response.diagnostic
            )
            torch.testing.assert_close(
                torch.tensor([row.predicted_value for row in response.results]).reshape(-1, 2),
                torch.tensor(expected),
            )
            bundle = torch.load(checkpoint, weights_only=True)
            bundle["node_feature_names"].reverse()
            torch.save(bundle, checkpoint)
            with self.assertRaises(ValueError):
                TorchBackend.load(checkpoint)

    def test_training_validation_test_and_model_package(self):
        backend, report = train_model(
            *self.splits,
            self.config,
            network=self.network,
            settings=TrainingConfig(epochs=2, patience=2, batch_size=4),
        )
        self.assertGreaterEqual(report["best_epoch"], 1)
        counts = [metrics["horizons"][0]["comparable_count"] for metrics in report["test"].values()]
        self.assertGreater(counts[0], 0)
        self.assertEqual(len(set(counts)), 1)
        self.assertEqual(backend.normalization, self.normalization)
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(self.temp_root.resolve()))
            path = Path(directory) / "trained.pt"
            save_bundle(
                path, backend.model, backend.normalization, self.config, training_summary=report
            )
            self.assertEqual(TorchBackend.load(path).config, self.config)

    def test_training_refuses_overlapping_flight_groups(self):
        with self.assertRaisesRegex(ValueError, "信息泄漏"):
            train_model(self.splits[0], self.splits[0], self.splits[2], self.config)

    def test_unified_graph_training_toml(self):
        path = Path(__file__).resolve().parents[1] / "config" / "formal_graph_training.toml"
        job = GraphTrainingJob.from_toml(path)
        self.assertEqual(job.model.history_steps, 10)
        self.assertEqual(job.model.forecast_steps, 3)
        self.assertEqual(job.network.hidden_dim, 64)
        self.assertEqual(job.network.link_dim, 64)
        self.assertEqual(job.network.lstm_hidden_dim, 64)
        self.assertEqual(job.network.graph_layers, 2)
        self.assertEqual(job.training.epochs, 30)
        self.assertEqual(job.stride_steps, 10)
        self.assertEqual(job.data.name, "index.json")


if __name__ == "__main__":
    unittest.main()
