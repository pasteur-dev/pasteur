"""Wraps the tqdm module so the same modules are used across the project.

Deals with vs code jupyter not supporting multiple progress bars."""

import functools
import io
import logging
import sys
from contextlib import contextmanager
from os import cpu_count, environ
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TextIO, TypeGuard, TypeVar

from rich import get_console
from tqdm import tqdm, trange

if TYPE_CHECKING:
    from multiprocessing.pool import Pool


X = TypeVar("X")
P = ParamSpec("P")
logger = logging.getLogger(__name__)

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

RICH_TRACEBACK_ARGS = {
    "show_locals": False,
    # "max_frames": 10,
    "suppress": ["kedro", "click"],
}

# Disable multiprocessing when debugging due to process launch debug overhead
MULTIPROCESS_ENABLE = not environ.get("_DEBUG", False)
IS_SUBPROCESS = False


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
    if IS_SUBPROCESS:
        """Disable subprocess pbars until a better solution."""
        disable = True
    else:
        active_pbars = sum(not pbar.disable for pbar in tqdm._instances)  # type: ignore
        disable = is_jupyter() and active_pbars >= JUPYTER_MAX_NEST
    return {
        "disable": disable,
        "colour": PBAR_COLOR,
        "bar_format": PBAR_FORMAT,
        "ncols": PBAR_JUP_NCOLS if is_jupyter() else None,
        "dynamic_ncols": not is_jupyter(),
        "ascii": True if is_jupyter() else None,
        "file": sys.stdout if is_jupyter() else sys.stderr,
    }


A = TypeVar("A", bound=Callable)


def limit_pbar_nesting(pbar_gen: A) -> A:
    """Prevent nesting too much on jupyter. This causes ugly gaps to be generated
    on vs code. Up to 2 progress bars work fine."""

    @functools.wraps(pbar_gen)
    def closure(*args, **kwargs):
        return pbar_gen(*args, **kwargs, **get_tqdm_args())

    return closure  # type: ignore


prange = limit_pbar_nesting(trange)
piter = limit_pbar_nesting(tqdm)


def _wrap_exceptions(
    fun: Callable[P, X], /, node_name: str, *args: P.args, **kwargs: P.kwargs
) -> X:
    set_node_name(node_name)
    try:
        return fun(*args, **kwargs)
    except Exception as e:
        get_console().print_exception(**RICH_TRACEBACK_ARGS)
        logger.error(
            f'Subprocess of "{get_node_name()}" failed with error:\n{type(e).__name__}: {e}'
        )
        raise RuntimeError("subprocess failed") from e


def _calc_worker(args):
    node_name, fun, base_args, chunk = args
    set_node_name(node_name)

    out = []
    for i, op in enumerate(chunk):
        try:
            args = {**base_args, **op} if base_args else op
            out.append(fun(**args))
        except Exception as e:
            get_console().print_exception(**RICH_TRACEBACK_ARGS)
            logger.error(
                f'Subprocess of "{get_node_name()}" at index {i} failed with error:\n{type(e).__name__}: {e}'
            )
            raise e

    return out


_pool: "Pool | None" = None


def _replace_loggers_with_queue(q):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.root)

    for l in loggers:
        l.propagate = True
        l.handlers = []
        l.level = logging.NOTSET

    from logging.handlers import QueueHandler

    logging.root.handlers.append(QueueHandler(q))


def _init_subprocess(lk, log_queue):
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if log_queue is not None:
        _replace_loggers_with_queue(log_queue)

    tqdm.set_lock(lk)

    global IS_SUBPROCESS
    IS_SUBPROCESS = True


def _get_pool():
    global _pool
    assert (
        _pool is not None
    ), f"Call `init_pool()` to create a process pool for the subprocessing nodes."
    return _pool


_node_name: Any | None = None


def set_node_name(name: str):
    global _node_name

    if _node_name is None:
        from threading import local

        _node_name = local()
        _node_name.name = name  # type: ignore
    else:
        _node_name.name = name


def get_node_name():
    assert _node_name and hasattr(
        _node_name, "name"
    ), "Node name has not been set, call `set_node_name()`."
    return _node_name.name


def init_pool(max_workers: int | None = None, log_queue=None):
    """Creates a shared process pool for all threads in this process.

    `max_workers` should be set based either on cores or on how many RAM GBs
    will be required by each process."""
    from multiprocessing import Pool

    global _pool

    if _pool is not None:
        logger.warning("Closing open pool...")
        _pool.terminate()

    lk = tqdm.get_lock()
    _pool = Pool(
        processes=max_workers,
        initializer=_init_subprocess,
        initargs=(lk, log_queue),
        # maxtasksperchild=1,
    )


def close_pool():
    if _pool is not None:
        _pool.terminate()


def process(fun: Callable[P, X], *args: P.args, **kwargs: P.kwargs) -> X:
    """Uses a separate process to complete this task, taken from the common pool."""
    if not MULTIPROCESS_ENABLE or IS_SUBPROCESS:
        return fun(*args, **kwargs)

    return _get_pool().apply(_wrap_exceptions, (fun, get_node_name(), *args), kwargs)  # type: ignore


def process_in_parallel(
    fun: Callable[..., X],
    per_call_args: list[dict],
    base_args: dict[str, Any] | None = None,
    min_chunk_size: int = 1,
    desc: str | None = None,
) -> list[X]:
    """Processes arguments in parallel using the common process pool and prints progress bar.

    Implements a custom form of chunk iteration, where `base_args` contains arguments
    with large size that are common in all function calls and `per_call_args` which
    change every iteration."""

    if (
        len(per_call_args) < 2 * min_chunk_size
        or not MULTIPROCESS_ENABLE
        or IS_SUBPROCESS
    ):
        out = []
        for op in piter(
            per_call_args, total=len(per_call_args), desc=desc, leave=False
        ):
            args = {**base_args, **op} if base_args else op
            out.append(fun(**args))

        return out

    import numpy as np

    cpus = cpu_count() or 64
    chunk_n = min(cpus * 5, len(per_call_args) // min_chunk_size)
    per_call_n = len(per_call_args) // chunk_n

    chunks = np.array_split(per_call_args, chunk_n)  # type: ignore

    args = []
    for chunk in chunks:
        args.append((get_node_name(), fun, base_args, chunk))

    pool = _get_pool()
    res = piter(
        pool.imap(_calc_worker, args),
        desc=f"{desc}, {per_call_n}/{len(per_call_args)} per it",
        leave=False,
        total=len(args),
    )

    out = []
    for sub_arr in res:
        out.extend(sub_arr)

    return out


def _is_console_logging_handler(handler) -> TypeGuard[logging.StreamHandler]:
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
    orig_streams: list[list[TextIO | None]] = []

    # Use stdout for jupyter to avoid coloring it red
    out_stream = sys.stdout if is_jupyter() else sys.stderr

    class PbarIO(io.StringIO):
        def write(self, text: str):
            tqdm.write(text[:-1], file=out_stream)

    # Swap rich logger
    pbar_stream = PbarIO()
    c = get_console()
    rich_fn = c.file
    c.file = pbar_stream

    try:
        for logger in loggers:
            # Swap standard loggers
            orig_streams.append([])
            for handler in logger.handlers:
                if _is_console_logging_handler(handler):
                    orig_streams[-1].append(handler.stream)  # type: ignore
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


__all__ = [
    "MULTIPROCESS_ENABLE",
    "piter",
    "prange",
    "process_in_parallel",
    "logging_redirect_pbar",
]
