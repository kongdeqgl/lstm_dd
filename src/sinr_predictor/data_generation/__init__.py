"""数据生成与转换：仿真导出转图数据，以及小规模合成数据。"""

from .material_graph import LocalFrame, convert_exports, sha256
from .synthetic import make_synthetic_dataset

__all__ = ["LocalFrame", "convert_exports", "make_synthetic_dataset", "sha256"]
