from multiprocessing.shared_memory import SharedMemory
from typing import Any, NamedTuple, TypeVar, cast

import numpy as np


class ArrayInfo(NamedTuple):
    shape: tuple[int, ...]
    dtype: Any
    ofs: int


A = TypeVar("A")


def map_to_memory(
    c: dict[A, list[np.ndarray]]
) -> tuple[SharedMemory, dict[A, list[ArrayInfo]]]:
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
    mem: SharedMemory,
    c: dict[A, list[ArrayInfo]],
    copy: bool = False,
    range: tuple[int, int] | None = None,
) -> dict[A, list[np.ndarray]]:
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
                a = a[range[0] : range[1]]
            if copy:
                a = a.copy()
            out[name].append(a)
    return out


def merge_memory(
    mem_data: list[tuple[SharedMemory, dict[A, list[ArrayInfo]], Any]], close: bool = True
) -> tuple[SharedMemory, dict[A, list[np.ndarray]]]:
    assert len(mem_data) > 0

    # Create instances of all col arrays
    cols = [load_from_memory(a, b) for a, b, _ in mem_data]

    # Calculate required size for final shared memory
    nbytes = 0
    for inst in cols:
        for col in inst.values():
            for height in col:
                nbytes += height.nbytes

    # Allocate memory even without arrays, to avoid optional logic, with min size 1
    out_mem = SharedMemory(name=None, create=True, size=max(nbytes, 1))
    out_info = {}

    # Start merging into new memory
    ofs = 0
    template = {k: len(v) for k, v in mem_data[0][1].items()}
    for name, max_height in template.items():
        out_info[name] = []
        for height in range(max_height):
            # Create new array using memory and append info
            new_rows = sum([len(inst[name][height]) for inst in cols])
            new_cols = cols[0][name][height].shape[1]
            new_shape = (new_rows, new_cols)
            dtype = cols[0][name][height].dtype
            arr = np.ndarray(new_shape, dtype, buffer=out_mem.buf, offset=ofs)
            out_info[name].append(ArrayInfo(new_shape, dtype, ofs))
            ofs += arr.nbytes

            # Fill in array with previous memory
            start = 0
            for inst in cols:
                col = inst[name][height]
                end = start + len(col)
                arr[start:end] = col
                start = end

    if close:
        del cols
        for mem, _, _ in mem_data:
            mem.close()

    return out_mem, out_info
