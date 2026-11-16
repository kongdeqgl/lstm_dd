"""按未来步统计误差、标签覆盖和算法可用率，全部指标恢复为 dB。"""

import math

from ..config import ModelConfig
from ..data.dataset import ForecastSample
from ..models.backends import PredictionBackend


def evaluate_samples(
    backend: PredictionBackend,
    samples: list[ForecastSample],
    config: ModelConfig,
    *,
    tolerance_db: float = 3.0,
) -> dict:
    if not math.isfinite(tolerance_db) or tolerance_db < 0:
        raise ValueError("容差必须为非负有限值，正式测试前应固定")
    errors = [[] for _ in range(config.forecast_steps)]
    label_counts = [0] * config.forecast_steps
    query_count = available_count = 0
    status_counts: dict[str, int] = {str(code): 0 for code in (0, 2, 3, 4)}
    for sample in samples:
        query_count += len(sample.windows)
        indices = sample.eligible_indices
        available_count += len(indices)
        for status in sample.statuses:
            status_counts[str(int(status))] += 1
        for row in sample.target_mask:
            for step, valid in enumerate(row):
                label_counts[step] += int(valid)
        if not indices:
            continue
        predictions = backend.predict(sample.eligible_windows, config)
        if len(predictions) != len(indices):
            raise ValueError("评估后端返回错误的查询数量")
        for index, prediction in zip(indices, predictions, strict=True):
            if len(prediction) != config.forecast_steps or any(
                not math.isfinite(value) for value in prediction
            ):
                raise ValueError("评估后端输出的步数错误或包含非有限值")
            for step, (value, target, valid) in enumerate(
                zip(
                    prediction,
                    sample.targets[index],
                    sample.target_mask[index],
                    strict=True,
                )
            ):
                if valid:
                    errors[step].append(value - target)
    horizons = []
    for step, values in enumerate(errors, start=1):
        count = len(values)
        horizons.append(
            {
                "step": step,
                "offset_ms": step * config.sampling_interval_ms,
                "label_count": label_counts[step - 1],
                "comparable_count": count,
                "mae_db": sum(abs(value) for value in values) / count if count else None,
                "rmse_db": math.sqrt(sum(value * value for value in values) / count)
                if count
                else None,
                "bias_db": sum(values) / count if count else None,
                "hit_rate": sum(abs(value) <= tolerance_db for value in values) / count
                if count
                else None,
            }
        )
    total_count = sum(len(values) for values in errors)
    squared = sum(value * value for values in errors for value in values)
    return {
        "query_count": query_count,
        "available_query_count": available_count,
        "availability": available_count / query_count if query_count else None,
        "status_counts": status_counts,
        "target_missing_ratio": 1 - sum(label_counts) / (query_count * config.forecast_steps)
        if query_count
        else None,
        "tolerance_db": tolerance_db,
        "overall_rmse_db": math.sqrt(squared / total_count) if total_count else None,
        "horizons": horizons,
    }
