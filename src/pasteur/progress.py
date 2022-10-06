"""Wraps the tqdm module so the same modules are used across the project.

Deals with vs code jupyter not supporting multiple progress bars."""

import functools
import io
import logging
import sys
from contextlib import contextmanager
from os import cpu_count, environ
from typing import Callable, TextIO, TypeVar

from tqdm import tqdm, trange
from tqdm.contrib.concurrent import process_map
from tqdm.std import tqdm as std_tqdm

# Jupyter doesn't support going up lines (moving cursor)
# This means up to 1 loading bar works
JUPYTER_MAX_NEST = 1
PBAR_COLOR = "blue"
PBAR_OFFSET = 11
PBAR_FORMAT = (" " * PBAR_OFFSET) + ">>>>>>>  {l_bar}{bar}{r_bar}"
# Exact number for notebooks rendered in github to use up the whole width
# Assumes a stripping github filter is used to remove the empty space (or time)
# at the start
PBAR_JUP_NCOLS = 135 + PBAR_OFFSET
PBAR_MIN_PIPE_LEN = 9

RICH_TRACEBACK_ARGS = {
    "show_locals": False,
    "max_frames": 10,
    "suppress": ["kedro", "click"],
}

# Disable multiprocessing when debugging due to process launch debug overhead
MULTIPROCESS_ENABLE = not environ.get("_DEBUG", False)

A = TypeVar("A")


def is_jupyter() -> bool:  # pragma: no cover
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


def get_tqdm_args():
    active_pbars = sum(not pbar.disable for pbar in std_tqdm._instances)
    disable = is_jupyter() and active_pbars >= JUPYTER_MAX_NEST
    return {
        "disable": disable,
        "colour": PBAR_COLOR,
        "bar_format": PBAR_FORMAT,
        "ncols": PBAR_JUP_NCOLS if is_jupyter() else None,
        "ascii": True if is_jupyter() else None,
        "file": sys.stdout if is_jupyter() else sys.stderr,
    }


def limit_pbar_nesting(pbar_gen: A) -> A:
    """Prevent nesting too much on jupyter. This causes ugly gaps to be generated
    on vs code. Up to 2 progress bars work fine."""

    @functools.wraps(pbar_gen)
    def closure(*args, **kwargs):
        return pbar_gen(*args, **kwargs, **get_tqdm_args())

    return closure


prange = limit_pbar_nesting(trange)
piter = limit_pbar_nesting(tqdm)


def calc_worker(args):
    fun, base_args, chunk = args
    out = []
    for op in chunk:
        args = {**base_args, **op} if base_args else op
        out.append(fun(**args))

    return out


X = TypeVar("X")


def process_in_parallel(
    fun: Callable[..., X],
    per_call_args: list[dict],
    base_args: dict[str, any] | None = None,
    min_chunk_size: int = 100,
    desc: str | None = None,
) -> list[X]:
    """Processes arguments in parallel using python's multiprocessing and prints progress bar.

    Task is split into chunks based on CPU cores and each process handles a chunk of
    calls before exiting."""
    import numpy as np

    if len(per_call_args) < 2 * min_chunk_size or not MULTIPROCESS_ENABLE:
        out = []
        for op in piter(per_call_args, total=len(per_call_args), leave=False):
            args = {**base_args, **op} if base_args else op
            out.append(fun(**args))

        return out

    chunk_n = min(cpu_count() * 5, len(per_call_args) // min_chunk_size)
    per_call_n = len(per_call_args) // chunk_n

    chunks = np.array_split(per_call_args, chunk_n)

    args = []
    for chunk in chunks:
        args.append((fun, base_args, chunk))

    res = process_map(
        calc_worker,
        args,
        desc=f"{desc}, {per_call_n}/{len(per_call_args)} per it",
        leave=False,
        tqdm_class=tqdm,
        **get_tqdm_args(),
    )
    out = []
    for sub_arr in res:
        out.extend(sub_arr)

    return out


def _is_console_logging_handler(handler):
    return isinstance(handler, logging.StreamHandler) and handler.stream in {
        sys.stdout,
        sys.stderr,
    }


@contextmanager
def logging_redirect_pbar():
    """ "Implementation of the logging_redirect_tqdm context manager that supports the rich handler."""
    from rich import get_console

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.root)
    orig_streams: list[list[TextIO]] = []

    # Use stdout for jupyter to avoid coloring it red
    out_stream = sys.stdout if is_jupyter() else sys.stderr

    class PbarIO(io.StringIO):
        def write(self, text: str):
            std_tqdm.write(text[:-1], file=out_stream)

    pbar_stream = PbarIO()
    rich_fn = None

    try:
        # Swap rich logger
        c = get_console()
        rich_fn = c.file
        c.file = pbar_stream

        for logger in loggers:
            # Swap standard loggers
            orig_streams.append([])
            for handler in logger.handlers:
                if _is_console_logging_handler(handler):
                    orig_streams[-1].append(handler.stream)
                    handler.setStream(pbar_stream)
                else:
                    orig_streams[-1].append(None)

        yield
    finally:
        for logger, orig_streams_logger in zip(loggers, orig_streams):
            for handler, stream in zip(logger.handlers, orig_streams_logger):
                if isinstance(handler, logging.StreamHandler) and stream is not None:
                    handler.setStream(stream)

        if rich_fn is not None:
            c.file = rich_fn


__all__ = ["piter", "prange", "process_in_parallel", "logging_redirect_pbar"]
