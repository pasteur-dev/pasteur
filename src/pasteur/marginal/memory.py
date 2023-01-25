from multiprocessing.shared_memory import SharedMemory
from typing import Any, NamedTuple, cast

import numpy as np

from ..attribute import Attributes, get_dtype, IdxValue
from ..utils import LazyFrame

class ArrayInfo(NamedTuple):
    shape: tuple[int]
    dtype: Any
    ofs: int

def allocate_memory(data: LazyFrame, attrs: Attributes, *, common: bool = False):
    assert data.partitioned, "Data is not partitioned"
    n, d = data.shape

    # Create write ranges for chunks + checks
    ranges = {}
    ofs = 0
    for name, chunk in data.items():
        cn, cd = chunk.shape
        assert d == cd, "Chunk has different number of columns than loaded data"
        
        ranges[name] = (ofs, ofs + cn)
        ofs += cn

    assert ofs == n, "Partitions have different length than merged load."
    
    # Create array infos and calculate nbytes
    ofs = 0
    info = {}
    for attr in attrs.values():
        # Skip allocation for attrs that don't have common
        if common and attr.common == 0:
            continue

        for name, col in attr.vals.items():
            col = cast(IdxValue, col)
            info[name] = []
            for height in range(col.height):
                shape = (n, )
                dtype = np.dtype(get_dtype(col.get_domain(height)))

                info[name].append(ArrayInfo(shape, dtype, ofs))
                ofs += dtype.itemsize * n
    
    nbytes = ofs
    mem = SharedMemory(name=None, create=True, size=max(nbytes, 1))
    return mem, info, ranges
    

def map_to_memory(c: dict[str, list[np.ndarray]]):
    nbytes = 0
    for arrs in c.values():
        for arr in arrs:
            nbytes += arr.nbytes

    # Allocate memory even without arrays, to avoid optional logic, with min size 1
    mem = SharedMemory(name=None, create=True, size=max(nbytes, 1))

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

    assert ofs == nbytes, "Shared memory buffer overflown"

    return mem, out


def load_from_memory(
    mem: SharedMemory, c: dict[str, list[ArrayInfo]], copy: bool = False, range: tuple[int, int] | None = None
) -> dict[str, list[np.ndarray]]:
    import mmap

    if hasattr(mem, "_mmap"):
        getattr(mem, "_mmap").madvise(mmap.MADV_SEQUENTIAL)

    out = {}
    for name, infos in c.items():
        out[name] = []
        for info in infos:
            shape, dtype, ofs = info
            # TODO: fix performance issue with shared_memory and remove .copy()
            a = np.ndarray(shape, dtype, buffer=mem.buf, offset=ofs)
            if range is not None:
                a = a[range[0]:range[1]]
            if copy:
                a = a.copy()
            out[name].append(a)
    return out
