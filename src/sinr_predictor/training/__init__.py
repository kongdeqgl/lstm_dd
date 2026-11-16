"""训练流程：样本训练、验证早停、独立测试与模型包导出。"""

__all__ = ["GraphTrainingJob", "TrainingConfig", "train_model"]


def __getattr__(name: str):
    # 仅访问训练入口时才加载 torch；单独使用评估模块仍可只依赖标准库。
    if name in __all__:
        from . import graph_train

        return getattr(graph_train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
