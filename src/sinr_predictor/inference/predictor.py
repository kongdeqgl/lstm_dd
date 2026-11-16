"""簇头在线推理服务编排。

一个簇长期复用一个 Predictor：持续接收当前时隙观测、维护历史缓存，
完成有效性判断后调用模型，并按外部接口格式返回多步预测结果。
"""

import math
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from ..config import ModelConfig
from ..data.eligibility import check_eligibility
from ..data.history import HistoryCache
from ..data.preprocess import build_query_window, build_snapshots
from ..models.backends import PersistenceBackend, PredictionBackend
from ..schemas import (
    FLOAT32_MAX,
    UINT64_MAX,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    RequestStatus,
    ResultStatus,
)


def system_time_ms() -> int:
    return time.time_ns() // 1_000_000


class Predictor:
    """维护单簇历史并提供线程安全的在线预测入口。"""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        backend: PredictionBackend | None = PersistenceBackend(),
        clock_ms: Callable[[], int] = system_time_ms,
    ) -> None:
        self.config = config if config is not None else ModelConfig()
        self.backend = backend
        self.history = HistoryCache()
        # clock_ms 仅供服务端和离线回放注入，不来自请求字段。
        self._clock_ms = clock_ms
        self._last_reference_ms: int | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, path: str | Path, **kwargs) -> "Predictor":
        return cls(ModelConfig.from_toml(path), **kwargs)

    @classmethod
    def from_model_bundle(cls, path: str | Path, *, device: str = "cpu", **kwargs) -> "Predictor":
        try:
            from ..models.neural_backend import TorchBackend
        except ImportError as error:
            raise RuntimeError("图 LSTM 模型需要安装算法依赖 torch") from error
        backend = TorchBackend.load(path, device=device)
        return cls(backend.config, backend=backend, **kwargs)

    def predict(self, payload: Mapping) -> PredictionResponse:
        """接收一个当前时隙批次；基准时间始终由簇头服务端确定。"""

        reference = self._clock_ms()
        if type(reference) is not int or not 0 <= reference <= UINT64_MAX:
            raise ValueError("服务端时钟必须返回 UINT64 毫秒时间戳")
        with self._lock:
            return self._predict_locked(payload, reference)

    def _finish(
        self,
        request_id: str,
        reference: int,
        status: RequestStatus,
        results: tuple[PredictionResult, ...] = (),
        diagnostic: str = "",
    ) -> PredictionResponse:
        generated = max(reference, self._clock_ms())
        return PredictionResponse(request_id, reference, generated, status, results, diagnostic)

    def _predict_locked(self, payload: Mapping, reference: int) -> PredictionResponse:
        config = self.config
        request_id = ""
        if isinstance(payload, Mapping):
            candidate = payload.get("request_id")
            if isinstance(candidate, str):
                try:
                    if (
                        candidate.strip()
                        and len(candidate.encode("utf-8")) <= 36
                        and not any(ord(c) < 32 for c in candidate)
                    ):
                        request_id = candidate
                except UnicodeError:
                    pass
        try:
            # 簇头可只使用本机接收时间：在线记录未提供观测时间时，
            # 以本次请求进入预测器的服务端基准时间作为有效观测时间。
            request = PredictionRequest.from_dict(
                payload, default_observation_time_ms=reference
            )
            request_id = request.request_id
            if self._last_reference_ms is not None and reference < self._last_reference_ms:
                raise ValueError("同一实例的请求时间必须非递减；独立回放请新建实例")
            if reference + config.forecast_steps * config.sampling_interval_ms > UINT64_MAX:
                raise ValueError("预测目标时间超出 UINT64 范围")
            for row in (*request.node_records, *request.relation_records):
                if row.observation_time_ms > reference:
                    raise ValueError("请求包含晚于服务端基准时间的未来观测")
            if any(row.source_id == row.destination_id for row in request.relation_records):
                raise ValueError("有向链路观测不能是自环")
            self.history.ingest(request, reference)
        except (ValueError, TypeError) as error:
            return self._finish(
                request_id, reference, RequestStatus.INVALID_REQUEST, diagnostic=str(error)
            )

        self._last_reference_ms = reference
        self.history.prune(reference, config.cache_retention_ms)
        if not request.query_relations:
            return self._finish(request_id, reference, RequestStatus.SUCCESS)
        if self.backend is None:
            return self._finish(
                request_id,
                reference,
                RequestStatus.MODEL_UNAVAILABLE,
                diagnostic="预测后端未加载",
            )

        # 先按规则构造十个历史时隙，再只把资格检查通过的链路送入模型。
        snapshots = build_snapshots(
            self.history, request.query_relations, reference, reference, config
        )
        windows = [
            build_query_window(self.history, snapshots, query, reference, reference, config)
            for query in request.query_relations
        ]
        statuses = [
            check_eligibility(window, self.history, reference, reference, config)
            for window in windows
        ]
        valid_indices = [
            index for index, status in enumerate(statuses) if status == ResultStatus.SUCCESS
        ]
        values_by_query: dict[int, list[float]] = {}
        if valid_indices:
            try:
                matrix = [
                    list(row)
                    for row in self.backend.predict(
                        [windows[index] for index in valid_indices],
                        config,
                    )
                ]
                if len(matrix) != len(valid_indices):
                    raise ValueError("后端输出的链路数不匹配")
                for index, row in zip(valid_indices, matrix, strict=True):
                    if len(row) != config.forecast_steps:
                        raise ValueError("后端输出的预测步数不匹配")
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or abs(value) > FLOAT32_MAX
                        for value in row
                    ):
                        raise ValueError("后端输出包含非法预测值")
                    values_by_query[index] = [float(value) for value in row]
            except Exception as error:
                # 后端故障不伪装成链路 SINR 低值；诊断信息留在进程内。
                return self._finish(
                    request_id,
                    reference,
                    RequestStatus.MODEL_UNAVAILABLE,
                    diagnostic=f"预测后端失败：{type(error).__name__}: {error}",
                )

        generated = max(reference, self._clock_ms())
        # 输出顺序固定为“查询链路优先、预测步次之”，与外部接口一致。
        results = []
        for index, query in enumerate(request.query_relations):
            for step in range(config.forecast_steps):
                target = reference + (step + 1) * config.sampling_interval_ms
                status = statuses[index]
                value = None
                if status == ResultStatus.SUCCESS:
                    if target <= generated:
                        status = ResultStatus.UNAVAILABLE
                    else:
                        value = values_by_query[index][step]
                results.append(
                    PredictionResult(
                        query.source_id,
                        query.destination_id,
                        target,
                        value,
                        status,
                    )
                )
        return PredictionResponse(
            request_id,
            reference,
            generated,
            RequestStatus.SUCCESS,
            tuple(results),
        )
