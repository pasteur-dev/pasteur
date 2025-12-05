"""This utility module provides constants and functions for multiprocessing
and progress monitoring in Pasteur.

In most cases, the functions and constants are simple wrappers around existing libraries.
The common use of the primitives in the codebase allows for using different
implementations in the future."""

import functools
import io
import logging
import re
import shutil
import sys
import time
from contextlib import contextmanager
from functools import reduce
from multiprocessing.pool import AsyncResult, Pool
from os import cpu_count, environ
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TextIO, TypeGuard, TypeVar

from rich import get_console
from tqdm import tqdm, trange

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager


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
PROGRESS_STEP_NS = 50_000_000

# Debug flag disables exception wrapping to print the correct stack trace so
# debugger stops at the top-most exception
DEBUG = bool(int(environ.get("_DEBUG", 0)))
MULTIPROCESS_ENABLE = bool(int(environ.get("_MULTIPROCESS", 1)))
IS_SUBPROCESS = False

CHECK_LEAKS = False


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


_jupyter = None


def is_jupyter() -> bool:
    # Avoid triggering raised exceptions
    global _jupyter

    if _jupyter is None:
        _jupyter = _is_jupyter()

    return _jupyter


def get_tqdm_args():
    if IS_SUBPROCESS:
        """Disable subprocess pbars until a better solution."""
        disable = True
    else:
        active_pbars = len(tqdm._instances)  # type: ignore
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
    fun: Callable[P, X],
    /,
    node_name: str,
    lock,
    first_error,
    *args: P.args,
    **kwargs: P.kwargs,
) -> X:
    set_node_name(node_name)
    try:
        res = fun(*args, **kwargs)

        if CHECK_LEAKS:
            # check for leaks after first execution
            from pasteur.utils.leaks import check, clear

            clear()
            a = fun(*args, **kwargs)
            del a
            check(f"Node {node_name} leaks")

        return res
    except Exception as e:
        # Use some logic to only print the first error
        if lock and first_error:
            with lock:
                _first_error = first_error.value
                first_error.value = False
        else:
            _first_error = True

        # raise original exception to to catch proper breakpoint
        if _first_error:
            _first_error = False
            if DEBUG:
                raise e
            get_console().print_exception(**RICH_TRACEBACK_ARGS)
            logger.error(
                f'Subprocess of "{get_node_name()}" failed with error:\n{type(e).__name__}: {e}'
            )
        raise RuntimeError("subprocess failed") from e


def _calc_worker(args):
    (
        node_name,
        progress_send,
        progress_lock,
        progress_first_error,
        initializer,
        fun,
        finalizer,
        base_args,
        chunk,
    ) = args
    set_node_name(node_name)

    u = 0
    try:
        ex = None
        if initializer is not None:
            try:
                base_args, chunk = initializer(base_args, chunk)
            except Exception as e:
                with progress_lock:
                    first_error = progress_first_error.value
                    progress_first_error.value = False
                if first_error:
                    if DEBUG:
                        raise e
                    get_console().print_exception(**RICH_TRACEBACK_ARGS)
                    logger.error(
                        f'Subprocess initialization of "{get_node_name()}" failed with error:\n{type(e).__name__}: {e}'
                    )
                raise RuntimeError("subprocess failed") from e

        last_update = time.perf_counter_ns()
        out = []
        u = 0
        for i, op in enumerate(chunk):
            try:
                args = {**base_args, **op} if base_args else op
                out.append(fun(**args))

                if CHECK_LEAKS:
                    # Run second so first run loads modules
                    from pasteur.utils.leaks import check, clear

                    clear()
                    a = fun(**args)
                    del a
                    check(f"Node {node_name}:{i} leaks")
            except Exception as e:
                with progress_lock:
                    first_error = progress_first_error.value
                    progress_first_error.value = False
                if first_error:
                    if DEBUG:
                        raise e
                    get_console().print_exception(**RICH_TRACEBACK_ARGS)
                    logger.error(
                        f'Subprocess of "{get_node_name()}" at index {i} failed with error:\n{type(e).__name__}: {e}'
                    )
                raise RuntimeError("subprocess failed") from e

            u += 1
            curr_time = time.perf_counter_ns()
            if curr_time - last_update > PROGRESS_STEP_NS:
                with progress_lock:
                    progress_send.send(u)
                last_update = curr_time
                u = 0

        if finalizer is not None:
            try:
                finalizer(base_args, chunk)
            except Exception as e:
                with progress_lock:
                    first_error = progress_first_error.value
                    progress_first_error.value = False
                if first_error:
                    if DEBUG:
                        raise e
                    get_console().print_exception(**RICH_TRACEBACK_ARGS)
                    logger.error(
                        f'Subprocess finalization of "{get_node_name()}" failed with error:\n{type(e).__name__}: {e}'
                    )
                raise RuntimeError("subprocess failed") from e

        return out
    finally:
        # Always close pipe. Raise special exception to hide stack trace
        if u != 0:
            with progress_lock:
                progress_send.send(u)
        progress_send.close()


_max_workers: int = 1
_pool: "tuple[Pool, SyncManager, Any] | None" = None


def get_max_workers() -> int:
    """Returns the maximum number of workers in the current process pool."""
    return _max_workers


def _logging_thread_fun(q):
    try:
        while True:
            record = q.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
    except EOFError:
        pass


def _replace_loggers_with_queue(q):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.root)

    for l in loggers:
        l.propagate = True
        l.handlers = []
        l.level = logging.NOTSET

    from logging.handlers import QueueHandler

    logging.root.handlers.append(QueueHandler(q))


def _init_subprocess(log_queue):
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if log_queue is not None:
        # Kedro installs rich logging when importing the following module
        # and messes with loggers. Import before replacing the loggers.
        import kedro.framework.project as _

        _replace_loggers_with_queue(log_queue)

    global IS_SUBPROCESS
    IS_SUBPROCESS = True


def _get_pool():
    global _pool
    if _pool is None:
        assert (
            MULTIPROCESS_ENABLE
        ), "Multiprocessing has been disabled. Preventing pool creation."
        logger.warning(
            "Launching a process pool implicitly. Use `init_pool()` to explicitly control pool creation."
        )
        init_pool()

    assert _pool is not None
    return _pool


def get_manager():
    """Returns the manager of the current process pool."""
    return _get_pool()[1]


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
    if _node_name and hasattr(_node_name, "name"):
        return _node_name.name
    return "UKN_NODE"


def close_pool():
    global _pool
    if _pool is None:
        return

    pool, _, log_queue = _pool
    log_queue.put(None)
    pool.terminate()
    _pool = None


def _init_pool(max_workers: int | None = None, refresh_processes: int | None = None):
    # from multiprocessing import Pool, Manager
    import threading
    from multiprocessing import get_context

    global _pool, _max_workers

    close_pool()

    ctx = get_context("spawn")
    manager = ctx.Manager()
    _max_workers = max_workers or cpu_count() or 1

    # set up logging handler for subprocesses
    log_queue = manager.Queue()
    lp = threading.Thread(target=_logging_thread_fun, args=(log_queue,))
    lp.start()

    pool = ctx.Pool(
        processes=max_workers,
        initializer=_init_subprocess,
        initargs=(log_queue,),
        maxtasksperchild=refresh_processes,
    )

    _pool = (pool, manager, log_queue)

    return _pool


class init_pool:
    def __init__(
        self, max_workers: int | None = None, refresh_processes: int | None = None
    ) -> None:
        """Creates a shared process pool for all threads in this process.

        `max_workers` should be set based either on cores or on how many RAM GBs
        will be required by each process.

        `log_queue` connects the subprocesses to the main process logger, see `pasteur.kedro.runner.parallel.py`

        `refresh_processes` sets `maxtasksperchild` for the pool, which
        prevents memory leaks from snowballing from node to node. However,
        due to additional imports every restart, it is slower."""
        _init_pool(max_workers, refresh_processes)

    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        close_pool()


def process(fun: Callable[P, X], *args: P.args, **kwargs: P.kwargs) -> X:
    """Uses a separate process to complete this task, taken from the common pool."""
    if not MULTIPROCESS_ENABLE or IS_SUBPROCESS:
        return fun(*args, **kwargs)

    pool, manager, _ = _get_pool()
    return pool.apply(_wrap_exceptions, (fun, get_node_name(), manager.Lock(), manager.Value("first_error", True), *args), kwargs)  # type: ignore


class AsyncResultStub(AsyncResult):
    def __init__(self, obj):
        super().__init__(None, None, None)  # type: ignore
        self.obj = obj

    def ready(self):
        return True

    def successful(self):
        return True

    def wait(self, timeout=None): ...

    def get(self, timeout=None):
        return self.obj


def process_async(
    fun: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
) -> AsyncResult:
    """Uses a separate process to complete this task, taken from the common pool."""
    if not MULTIPROCESS_ENABLE or IS_SUBPROCESS:
        return AsyncResultStub(fun(*args, **kwargs))

    pool, manager, _ = _get_pool()
    return pool.apply_async(_wrap_exceptions, (fun, get_node_name(), manager.Lock(), manager.Value("first_error", True), *args), kwargs)  # type: ignore


def process_in_parallel(
    fun: Callable[..., X],
    per_call_args: list[dict],
    base_args: dict[str, Any] | None = None,
    min_chunk_size: int = 1,
    desc: str | None = None,
    max_worker_mult: int = 1,
    initializer: Callable | None = None,
    finalizer: Callable[..., None] | None = None,
) -> list[X]:
    """Processes arguments in parallel using the common process pool and prints progress bar.

    Implements a custom form of chunk iteration, where `base_args` contains arguments
    with large size that are common in all function calls and `per_call_args` which
    change every iteration."""

    from multiprocessing import Lock, Pipe

    if (
        # len(per_call_args) < 2 * min_chunk_size
        not MULTIPROCESS_ENABLE
        or IS_SUBPROCESS
    ):
        if initializer is not None:
            base_args, per_call_args = initializer(base_args, per_call_args)

        out = []
        for args in piter(
            per_call_args,
            desc=desc,
            leave=False,
        ):
            kwargs = args.copy()
            if base_args:
                kwargs.update(base_args)
            res = _wrap_exceptions(fun, get_node_name(), None, None, **kwargs)
            out.append(res)

        if finalizer is not None:
            finalizer(base_args, per_call_args)

        return out

    pool, manager, _ = _get_pool()
    progress_recv, progress_send = Pipe(duplex=False)
    progress_lock = manager.Lock()
    progress_first_error = manager.Value("first_error", True)

    n_tasks = len(per_call_args)
    if n_tasks == 0:
        return []

    chunk_n_suggestion = min(
        max_worker_mult * _max_workers, (n_tasks - 1) // min_chunk_size + 1
    )
    chunk_len = (n_tasks - 1) // chunk_n_suggestion + 1
    chunk_n = (n_tasks - 1) // chunk_len + 1
    chunks = [
        per_call_args[chunk_len * j : min(chunk_len * (j + 1), n_tasks)]
        for j in range(chunk_n)
    ]

    args = []
    for chunk in chunks:
        args.append(
            (
                get_node_name(),
                progress_send,
                progress_lock,
                progress_first_error,
                initializer,
                fun,
                finalizer,
                base_args,
                chunk,
            )
        )

    res = pool.map_async(
        _calc_worker, args, error_callback=lambda _: progress_send.close()
    )

    pbar = piter(desc=desc, leave=False, total=n_tasks)
    n = 0
    while not res.ready():
        try:
            u = progress_recv.recv()
            n += u
            pbar.update(u)
            if n == n_tasks:
                break
        except Exception:
            break

    out = []
    for sub_arr in res.get():
        out.extend(sub_arr)

    progress_send.close()
    progress_recv.close()
    pbar.close()
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

    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.root)
    orig_streams: list[list[TextIO | None]] = []

    # Use stdout for jupyter to avoid coloring it red
    out_stream = sys.stdout if is_jupyter() else sys.stderr

    class PbarIO(io.StringIO):
        rmtext = ""

        def write(self, text: str):
            if self.rmtext:
                queue_rm = self.rmtext
                self.rmtext = ""
            else:
                queue_rm = ""

            if ":ephemeral:" in text:
                cleaned = text.replace(":ephemeral:", "")
                term_size = shutil.get_terminal_size(fallback=(80, 24))
                term_w = term_size.columns
                active_pbars = len(tqdm._instances) if hasattr(tqdm, "_instances") else 0  # type: ignore[attr-defined]
                term_h = max(1, term_size.lines - active_pbars)

                def wrapped_rows(line: str, _extra: int = 0) -> int:
                    vlen = len(ansi_re.sub("", line))
                    return max(1, (vlen + term_w - 1 + _extra) // term_w)

                # keep only the trailing rows that fit on screen, with the
                # first line for context
                lines = cleaned.splitlines() or [""]
                kept: list[str] = []
                used = wrapped_rows(lines[0])

                leftover = False
                for line in reversed(lines[1:]):
                    need = wrapped_rows(line)
                    if used + need > term_h:
                        leftover = True
                        break
                    kept.append(line)
                    used += need
                kept.reverse()
                kept.insert(0, lines[0])

                if leftover:
                    kept[0] = kept[0]

                text = "\n".join(kept) if kept else ""
                self.rmtext = "\033[F\033[2K" * max(used, 1)

            tqdm.write(queue_rm + text[:-1], file=out_stream)

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
    "process",
    "process_async",
    "process_in_parallel",
    "logging_redirect_pbar",
    "init_pool",
    "close_pool",
    "get_manager",
    "set_node_name",
    "get_node_name",
    "reduce",
]
