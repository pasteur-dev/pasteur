from multiprocessing.shared_memory import SharedMemory
from typing import Any, NamedTuple

import numpy as np


class ArrayInfo(NamedTuple):
    shape: tuple[int]
    dtype: Any
    ofs: int


def map_to_memory(c: dict[str, list[np.ndarray]]):
    mem = 0
    for arrs in c.values():
        for arr in arrs:
            mem += arr.nbytes

    mem = SharedMemory(name=None, create=True, size=mem)

    ofs = 0
    out = {}
    for name, arrs in c.items():
        out[name] = []
        for arr in arrs:
            # Create info object
            shape = arr.shape
            dtype = arr.dtype
            info = ArrayInfo(shape, dtype, ofs)

            # Copy Data
            buf = np.ndarray(shape, dtype, buffer=mem.buf, offset=ofs)
            np.copyto(buf, arr, casting="no")

            out[name].append(info)
            ofs += buf.nbytes

    return mem, out


def load_from_memory(
    mem: SharedMemory, c: dict[str, list[ArrayInfo]]
) -> dict[str, list[np.ndarray]]:
    out = {}
    for name, infos in c.items():
        out[name] = []
        for info in infos:
            shape, dtype, ofs = info
            out[name].append(np.ndarray(shape, dtype, buffer=mem.buf, offset=ofs))
    return out