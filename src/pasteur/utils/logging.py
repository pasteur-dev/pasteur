from io import StringIO
from logging import Handler, getLevelName
from os import path
from time import time
from weakref import WeakSet

import mlflow


class MlflowHandler(Handler):
    """
    A handler class based on `StreamHandler` and `MemoryHandler`, which
    buffers the records using a StringIO and dumps that StringIO every
    N records into an mlflow artifact.

    @Warning: currently doesn't have a size limit
    """

    terminator = "\n"
    _handlers: WeakSet["MlflowHandler"] = WeakSet()

    @staticmethod
    def close_all():
        """Call this function to close the mlflow handlers before ending a run
        if you don't plan to reuse them."""
        for handler in MlflowHandler._handlers:
            handler.close()

    @staticmethod
    def reset_all():
        """Flushes and cleans up all handlers so that they can be used for a new run."""
        for handler in MlflowHandler._handlers:
            handler.reset()

    @staticmethod
    def flush_all():
        for handler in MlflowHandler._handlers:
            handler.flush()

    def __init__(
        self,
        name: str = "user",
        logdir: str = "_logs",
        interval: int | str = 5,
    ):
        Handler.__init__(self)
        self._handlers.add(self)

        self._mlflow = None
        self.name = name
        self.logdir = logdir

        self.interval = interval
        self.last_sent = time()

        self.stream = StringIO()

    def flush(self):
        self.acquire()
        try:
            if mlflow.active_run() is not None:
                mlflow.log_text(
                    self.stream.getvalue(), path.join(self.logdir, f"{self.name}.log")
                )
                self.last_sent = time()
        finally:
            self.release()

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # issue 35046: merged two stream.writes into one.
            stream.write(msg + self.terminator)
            if time() - self.last_sent > self.interval:
                self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self.flush()
        finally:
            super().close()

    def reset(self):
        """Flushes and removes the current buffer of the handler, allowing it
        to be reused for a new run."""
        self.acquire()
        try:
            self.flush()
            self.steam = StringIO()
        finally:
            self.release()

    def __repr__(self):
        level = getLevelName(self.level)
        return "<Mlflow handler %s(%s)>" % (self.name, level)
