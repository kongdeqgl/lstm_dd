"""统一预测协议与基线：最近观测保持、历史均值；供训练评估和推理共用。"""

from collections.abc import Sequence
from typing import Protocol

from ..config import ModelConfig
from ..data.preprocess import QueryWindow


class PredictionBackend(Protocol):
    def predict(
        self,
        windows: Sequence[QueryWindow],
        config: ModelConfig,
    ) -> Sequence[Sequence[float]]:
        """输入 Q 条合格链路，输出 [Q, forecast_steps]，单位 dB。"""
        ...


class PersistenceBackend:
    """最近观测保持基线，不包含已训练的神经网络或精度承诺。"""

    def predict(
        self,
        windows: Sequence[QueryWindow],
        config: ModelConfig,
    ) -> list[list[float]]:
        predictions = []
        for window in windows:
            latest = next(
                entry.link_quality
                for entry in reversed(window.entries)
                if entry is not None and entry.link_quality is not None
            )
            predictions.append([latest] * config.forecast_steps)
        return predictions


class HistoryMeanBackend:
    """只平均窗口内的新有效观测，短时保持值不重复计数。"""

    def predict(
        self,
        windows: Sequence[QueryWindow],
        config: ModelConfig,
    ) -> list[list[float]]:
        predictions = []
        for window in windows:
            values = [
                entry.link_quality
                for entry, observed in zip(
                    window.entries,
                    window.observed_mask,
                    strict=True,
                )
                if observed and entry is not None and entry.link_quality is not None
            ]
            if not values:
                raise ValueError("历史均值基线没有新有效观测")
            predictions.append([sum(values) / len(values)] * config.forecast_steps)
        return predictions
