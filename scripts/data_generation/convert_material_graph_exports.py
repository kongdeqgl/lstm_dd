"""把算料生成 1.5 的完整输出转换为分片图训练数据。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sinr_predictor.data_generation.material_graph import convert_exports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="转换算料生成 1.5 为图网络训练数据")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke-groups",
        type=int,
        default=0,
        help="另生成 smoke_index.json，从前 N 个场景族各取一次运行",
    )
    args = parser.parse_args(argv)
    try:
        metadata = convert_exports(
            args.run_root, args.output, smoke_groups=args.smoke_groups
        )
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "run_count": metadata["run_count"],
                    "training_group_count": metadata["training_group_count"],
                    "frame_count": metadata["frame_count"],
                    "valid_relation_count": metadata["valid_relation_count"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        print(f"转换失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
