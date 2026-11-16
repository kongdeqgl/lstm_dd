"""图网络 + LSTM 的正式训练流程。

本模块只负责训练编排：读取统一 TOML、构造样本、训练、验证早停、
独立测试和模型包导出。数据转换与在线推理分别位于其他子包。
"""

import argparse
import json
import math
import random
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import torch

from ..config import ModelConfig
from ..data.dataset import ForecastSample, build_samples, load_episodes, split_episodes
from ..data.normalization import Normalization
from ..data.tensors import tensorize_windows
from ..models.backends import HistoryMeanBackend, PersistenceBackend
from ..models.graph_lstm import GraphLSTM, NetworkConfig, masked_mse
from ..models.neural_backend import TorchBackend, save_bundle
from .evaluate import evaluate_samples


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 8
    gradient_clip: float = 1.0
    min_delta_db: float = 0.00001
    seed: int = 42
    device: str = "cpu"
    threads: int = 1
    tolerance_db: float = 3.0

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} 必须为正整数")
        for name in (
            "learning_rate",
            "gradient_clip",
            "weight_decay",
            "min_delta_db",
            "tolerance_db",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} 必须是非负有限值")
        if not self.learning_rate or not self.gradient_clip or type(self.seed) is not int:
            raise ValueError("学习率、裁剪阈值须为正，seed 须为整数")


@dataclass(frozen=True, slots=True)
class GraphTrainingJob:
    """一个可复现的图模型训练任务；路径相对 TOML 文件解析。"""

    data: Path
    output: Path
    model: ModelConfig
    network: NetworkConfig
    training: TrainingConfig
    stride_steps: int = 1

    def __post_init__(self) -> None:
        if type(self.stride_steps) is not int or self.stride_steps < 1:
            raise ValueError("stride_steps 必须为正整数")

    @classmethod
    def from_toml(cls, path: str | Path) -> "GraphTrainingJob":
        """读取一次完整训练任务，并将相对路径锚定到配置文件目录。"""

        source = Path(path).resolve()
        with source.open("rb") as stream:
            document = tomllib.load(stream)
        sections = {"paths", "model", "network", "training"}
        if set(document) != sections or any(
            not isinstance(document[name], dict) for name in sections
        ):
            raise ValueError("统一训练配置必须仅包含 paths、model、network、training")

        paths = document["paths"]
        if set(paths) != {"data", "output"} or any(
            not isinstance(paths[name], str) or not paths[name].strip()
            for name in ("data", "output")
        ):
            raise ValueError("[paths] 必须仅包含非空字符串 data 和 output")

        def checked(section: str, allowed: set[str]) -> dict:
            values = document[section]
            unknown = values.keys() - allowed
            if unknown:
                raise ValueError(
                    f"[{section}] 含未知配置项：{', '.join(sorted(unknown))}"
                )
            return values

        model_values = checked("model", {field.name for field in fields(ModelConfig)})
        network_values = checked("network", {field.name for field in fields(NetworkConfig)})
        training_values = dict(
            checked(
                "training",
                {field.name for field in fields(TrainingConfig)} | {"stride_steps"},
            )
        )
        stride = training_values.pop("stride_steps", 1)

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return (
                candidate.resolve()
                if candidate.is_absolute()
                else (source.parent / candidate).resolve()
            )

        return cls(
            data=resolve(paths["data"]),
            output=resolve(paths["output"]),
            model=ModelConfig(**model_values),
            network=NetworkConfig(**network_values),
            training=TrainingConfig(**training_values),
            stride_steps=stride,
        )


def _supervision(sample: ForecastSample, normalization: Normalization, device: torch.device):
    indices = sample.eligible_indices
    mask = torch.tensor(
        [sample.target_mask[index] for index in indices], dtype=torch.bool, device=device
    )
    targets = torch.tensor(
        [
            [
                (value - normalization.target.mean[0]) / normalization.target.scale[0]
                if valid
                else 0.0
                for value, valid in zip(
                    sample.targets[index], sample.target_mask[index], strict=True
                )
            ]
            for index in indices
        ],
        dtype=torch.float32,
        device=device,
    )
    return targets, mask


def train_model(
    training: list[ForecastSample],
    validation: list[ForecastSample],
    test: list[ForecastSample],
    config: ModelConfig,
    *,
    network: NetworkConfig | None = None,
    settings: TrainingConfig | None = None,
    progress=None,
) -> tuple[TorchBackend, dict]:
    """训练模型并返回可直接推理的后端与完整评估报告。"""

    settings = settings or TrainingConfig()
    network = network or NetworkConfig()
    group_sets = [{sample.episode_id for sample in split} for split in (training, validation, test)]
    if any(group_sets[left] & group_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("训练、验证、测试之间存在相同飞行任务，拒绝信息泄漏")
    # 训练、验证、测试必须按完整场景划分，不能让同一飞行任务跨集合泄漏。
    useful = [
        sample
        for sample in training
        if any(any(sample.target_mask[index]) for index in sample.eligible_indices)
    ]
    if not useful:
        raise ValueError("训练集没有同时具备有效历史和真实未来标签的样本")
    torch.set_num_threads(settings.threads)
    torch.manual_seed(settings.seed)
    rng = random.Random(settings.seed)
    # 标准化统计量只从训练集拟合，并随 model.pt 一并保存。
    normalization = Normalization.fit(training, config)
    model = GraphLSTM(config.forecast_steps, network).to(settings.device)
    backend = TorchBackend(model, normalization, config, device=settings.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    # 先验证标签可用性，不能拿训练误差代替验证模型选择。
    initial_validation = evaluate_samples(
        backend, validation, config, tolerance_db=settings.tolerance_db
    )
    if initial_validation["overall_rmse_db"] is None:
        raise ValueError("验证集没有可比较的未来标签")
    best_rmse, best_state, best_epoch, stale_epochs = math.inf, None, 0, 0
    history = []
    started = time.perf_counter()
    # 每轮训练后都在验证集评估，连续无提升时提前停止。
    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = list(useful)
        rng.shuffle(order)
        squared_sum, target_count = 0.0, 0
        for start in range(0, len(order), settings.batch_size):
            optimizer.zero_grad(set_to_none=True)
            losses, batch_count = [], 0
            # 每个簇历史图独立前向；累积同一 mini-batch 的有效目标后再反向。
            for sample in order[start : start + settings.batch_size]:
                prediction = model(
                    tensorize_windows(
                        sample.eligible_windows,
                        normalization,
                        config,
                        backend.device,
                    )
                )
                target, mask = _supervision(sample, normalization, backend.device)
                loss = masked_mse(prediction, target, mask)
                if loss is None:
                    continue
                count = int(mask.sum().item())
                losses.append(loss * count)
                batch_count += count
            if not batch_count:
                continue
            loss = torch.stack(losses).sum() / batch_count
            if not bool(torch.isfinite(loss)):
                raise ValueError("训练损失非有限，请检查输入与标准化参数")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            squared_sum += float(loss.detach()) * batch_count
            target_count += batch_count
        validation_metrics = evaluate_samples(
            backend, validation, config, tolerance_db=settings.tolerance_db
        )
        rmse = validation_metrics["overall_rmse_db"]
        entry = {
            "epoch": epoch,
            "train_mse_normalized": squared_sum / target_count,
            "validation_rmse_db": rmse,
        }
        history.append(entry)
        if progress:
            progress(entry)
        if rmse < best_rmse - settings.min_delta_db:
            best_rmse, best_epoch, stale_epochs = rmse, epoch, 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if stale_epochs >= settings.patience:
            break
    assert best_state is not None
    # 最终导出的不是最后一轮，而是验证集表现最好的权重。
    model.load_state_dict(best_state)
    model.eval()
    # 测试集只在验证选模结束后评价，不参与拟合统计量或早停。
    report = {
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "model_config": asdict(config),
        "network_config": asdict(network),
        "training_config": asdict(settings),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "episode_ids": {
            name: sorted(ids)
            for name, ids in zip(("train", "validation", "test"), group_sets, strict=True)
        },
        "sample_counts": {"train": len(training), "validation": len(validation), "test": len(test)},
        "best_epoch": best_epoch,
        "best_validation_rmse_db": best_rmse,
        "initial_validation_rmse_db": initial_validation["overall_rmse_db"],
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "validation": evaluate_samples(
            backend, validation, config, tolerance_db=settings.tolerance_db
        ),
        "test": {
            name: evaluate_samples(candidate, test, config, tolerance_db=settings.tolerance_db)
            for name, candidate in (
                (
                    {
                        "graph": "graph_lstm",
                        "endpoints": "endpoints_lstm",
                        "quality": "quality_lstm",
                    }[network.spatial_mode],
                    backend,
                ),
                ("persistence", PersistenceBackend()),
                ("history_mean", HistoryMeanBackend()),
            )
        },
    }
    return backend, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="训练图空间编码 + 链路级 LSTM，需 Python 3.12 和 torch"
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        help="统一 TOML：数据路径、模型窗口、网络结构和训练参数",
    )
    parser.add_argument("--data", type=Path, help="训练数据；可覆盖统一配置")
    parser.add_argument("--config", type=Path, help="旧式模型窗口 TOML")
    parser.add_argument("--output", type=Path, help="输出目录；可覆盖统一配置")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="同时覆盖节点/边、链路和 LSTM 隐藏维度",
    )
    parser.add_argument("--graph-layers", type=int)
    parser.add_argument("--lstm-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--spatial-mode", choices=("graph", "endpoints", "quality")
    )
    parser.add_argument("--stride-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--tolerance-db", type=float)
    args = parser.parse_args(argv)
    try:
        job = GraphTrainingJob.from_toml(args.training_config) if args.training_config else None
        data = args.data.resolve() if args.data else (job.data if job else None)
        if data is None:
            parser.error("必须提供 --training-config 或 --data")
        output = (
            args.output.resolve()
            if args.output
            else (job.output if job else Path("outputs/graph_lstm").resolve())
        )
        config = (
            ModelConfig.from_toml(args.config)
            if args.config
            else job.model
            if job
            else replace(ModelConfig(), model_version="graph-lstm-v1")
        )
        network = job.network if job else NetworkConfig()
        if args.hidden_dim is not None:
            network = replace(
                network,
                hidden_dim=args.hidden_dim,
                link_dim=args.hidden_dim,
                lstm_hidden_dim=args.hidden_dim,
            )
        network_overrides = {
            "graph_layers": args.graph_layers,
            "lstm_layers": args.lstm_layers,
            "dropout": args.dropout,
            "spatial_mode": args.spatial_mode,
        }
        network = replace(
            network,
            **{name: value for name, value in network_overrides.items() if value is not None},
        )
        settings = job.training if job else TrainingConfig()
        setting_overrides = {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": args.device,
            "threads": args.threads,
            "tolerance_db": args.tolerance_db,
        }
        settings = replace(
            settings,
            **{name: value for name, value in setting_overrides.items() if value is not None},
        )
        stride_steps = (
            args.stride_steps
            if args.stride_steps is not None
            else job.stride_steps
            if job
            else 1
        )
        splits = split_episodes(load_episodes(data), seed=settings.seed)
        samples = [build_samples(split, config, stride_steps=stride_steps) for split in splits]
        backend, report = train_model(
            *samples,
            config,
            network=network,
            settings=settings,
            progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True),
        )
        report["dataset_path"] = str(data)
        report["stride_steps"] = stride_steps
        if args.training_config:
            report["training_config_path"] = str(args.training_config.resolve())
        output.mkdir(parents=True, exist_ok=True)
        save_bundle(
            output / "model.pt",
            backend.model,
            backend.normalization,
            config,
            training_summary=report,
        )
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"模型：{output / 'model.pt'}")
        print(f"报告：{output / 'report.json'}")
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(f"训练失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
