"""Wraps the tqdm module so the same modules are used across the project.

Deals with vs code jupyter not supporting multiple progress bars."""

from os import cpu_count

import numpy as np
from tqdm import tqdm, trange
from tqdm.asyncio import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.std import tqdm as std_tqdm

# VS code jupyter extension doesn't support going up lines (moving cursor)
# This means up to 1 loading bar works
# However, if there's another one with leave=True it can also be shown
# Any more than that and it causes garbage to build up at the output.
JUPYTER_MAX_NEST = 2


def _is_jupyter() -> bool:  # pragma: no cover
    """Check if we're running in a Jupyter notebook.

    taken from rich package."""
    try:
        get_ipython  # type: ignore[name-defined]
    except NameError:
        return False
    ipython = get_ipython()  # type: ignore[name-defined]
    shell = ipython.__class__.__name__
    if "google.colab" in str(ipython.__class__) or shell == "ZMQInteractiveShell":
        return True  # Jupyter notebook or qtconsole
    elif shell == "TerminalInteractiveShell":
        return False  # Terminal running IPython
    else:
        return False  # Other type (?)


def limit_pbar_nesting(pbar_gen: callable):
    """Prevent nesting too much on jupyter. This causes ugly gaps to be generated
    on vs code. Up to 2 progress bars work fine."""
    if not _is_jupyter():
        return pbar_gen

    def closure(*args, **kwargs):
        active_pbars = sum(not pbar.disable for pbar in std_tqdm._instances)
        return pbar_gen(
            *args,
            **kwargs,
            disable=active_pbars >= JUPYTER_MAX_NEST,
        )

    return closure


logging_redirect_pbar = logging_redirect_tqdm
prange = limit_pbar_nesting(trange)
piter = limit_pbar_nesting(tqdm)


def calc_worker(args):
    fun, base_args, chunk = args
    out = []
    for op in chunk:
        args = {**base_args, **op} if base_args else op
        out.append(fun(**args))

    return out


def process_in_parallel(
    fun: callable,
    per_call_args: list[dict],
    base_args: dict[str, any] | None = None,
    min_chunk_size: int = 100,
    desc: str | None = None,
):
    """Processes arguments in parallel using python's multiprocessing and prints progress bar.

    Task is split into chunks based on CPU cores and each process handles a chunk of
    calls before exiting."""
    if len(per_call_args) < 2 * min_chunk_size:
        return calc_worker((fun, base_args, per_call_args))

    chunk_n = min(cpu_count() * 5, len(per_call_args) // min_chunk_size)
    per_call_n = len(per_call_args) // chunk_n

    chunks = np.array_split(per_call_args, chunk_n)

    # Test for disabling too many nested pbars in jupyter
    active_pbars = sum(not pbar.disable for pbar in std_tqdm._instances)
    disable = _is_jupyter() and active_pbars >= JUPYTER_MAX_NEST

    args = []
    for chunk in chunks:
        args.append((fun, base_args, chunk))

    res = process_map(
        calc_worker,
        args,
        desc=f"{desc}, {per_call_n}/{len(per_call_args)} per it",
        leave=False,
        tqdm_class=tqdm,
        disable=disable,
    )
    out = []
    for sub_arr in res:
        out.extend(sub_arr)

    return out


__all__ = ["piter", "prange", "process_in_parallel", "logging_redirect_pbar"]
