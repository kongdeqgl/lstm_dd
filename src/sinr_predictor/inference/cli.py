"""正式模型的 JSON 文件推理入口；回放时间不是在线请求字段。"""

import argparse
import json
import sys
from pathlib import Path

from ..config import ModelConfig
from ..demo import DEMO_REFERENCE_TIME_MS, make_demo_request
from ..schemas import RequestStatus
from .predictor import Predictor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Python 3.12 有向链路 SINR 预测基线")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="运行固定时间的合成数据示例")
    source.add_argument("--request", type=Path, help="请求 JSON 文件，默认采用实际接收时间")
    parser.add_argument("--config", type=Path, help="模型参数 TOML；缺省使用内置示例参数")
    parser.add_argument("--model", type=Path, help="图 LSTM 模型包；与 --config 互斥")
    parser.add_argument("--device", default="cpu", help="神经网络运行设备")
    parser.add_argument("--replay-time-ms", type=int, help="仅用于离线回放的基准时间")
    parser.add_argument("--output", type=Path, help="响应 JSON 保存位置；缺省输出到终端")
    args = parser.parse_args(argv)
    if args.model and args.config:
        parser.error("--model 自带预处理参数，不能同时指定 --config")
    try:
        predictor = (
            Predictor.from_model_bundle(args.model, device=args.device) if args.model else None
        )
        config = (
            predictor.config
            if predictor
            else (ModelConfig.from_toml(args.config) if args.config else ModelConfig())
        )
        if args.demo:
            if args.replay_time_ms is not None:
                parser.error("--demo 使用固定示例时间，不能同时设置 --replay-time-ms")
            replay_time = DEMO_REFERENCE_TIME_MS
            payload = make_demo_request(replay_time, config)
        else:
            replay_time = args.replay_time_ms
            payload = json.loads(args.request.read_text(encoding="utf-8-sig"))
        if predictor is None:
            predictor = (
                Predictor(config, clock_ms=lambda: replay_time)
                if replay_time is not None
                else Predictor(config)
            )
        elif replay_time is not None:
            predictor = Predictor(config, backend=predictor.backend, clock_ms=lambda: replay_time)
        response = predictor.predict(payload)
        text = json.dumps(response.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        if response.diagnostic:
            print(response.diagnostic, file=sys.stderr)
        return {
            RequestStatus.SUCCESS: 0,
            RequestStatus.INVALID_REQUEST: 2,
            RequestStatus.MODEL_UNAVAILABLE: 3,
        }[response.status_code]
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"运行失败：{error}", file=sys.stderr)
        return 2
