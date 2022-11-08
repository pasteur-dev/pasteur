from ...utils.progress import MULTIPROCESS_ENABLE

if MULTIPROCESS_ENABLE:
    from .parallel import SimpleParallelRunner

    SimpleRunner = SimpleParallelRunner
else:
    from .sequential import SimpleSequentialRunner

    SimpleRunner = SimpleSequentialRunner

__all__ = [
    "SimpleRunner"
]