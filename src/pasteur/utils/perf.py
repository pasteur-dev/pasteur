import logging


logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks code performance by hooking to Pasteur internals and logs to mlflow.

    Works similar to `logging` loggers. Grab the tracker with the appropriate name
    with `get()`, log various operations in the module by calling
    `start()`, `stop()`, and have them logged as `<perf>.<tracker>.<name>`.

    For multistep tracking, call `ensemble()` with the name of the multistep
    tracking operation and the steps it depends on.

    Example:
    ```python
    t = PerformanceTracker.get("synth")
    t.ensemble("total", "bake", "fit", "sample")

    t.start("bake")
    # bake...
    t.stop("bake")

    t.start("fit")
    # fit...
    t.stop("fit")

    t.start("sample")
    # sample...
    t.stop("sample")
    ```

    Logs:
    ```
    perf.synth.bake = ...
    perf.synth.fit = ...
    perf.synth.sample = ...
    perf.synth.total = ...
    ```

    """

    _trackers: dict[str, "PerformanceTracker"] = {}

    def __init__(self) -> None:
        from time import time_ns

        self.starts: dict[str, int] = {}
        self.stops: dict[str, int] = {}
        self.ensembles: dict[str, list[int]] = {}
        self._log_to_file = False
        self.gpu = False
        self.time_ns = time_ns

    def log_to_file(self):
        self._log_to_file = True

    def use_gpu(self):
        self.gpu = True

    def start(self, name: str):
        self.starts[name] = self.time_ns()

    def stop(self, name: str):
        self.stops[name] = self.time_ns()

    def ensemble(self, name: str, *names: str | list[str]):
        if len(names) == 1 and not isinstance(names[0], str):
            names = names[0]  # type: ignore
        self.ensembles[name] = names  # type: ignore

    def get_log_dict(self):
        out = {}
        runs = self.starts.keys()

        for run in runs:
            # should not crash for missing runs, but it should also be obvious
            time = self.stops.get(run, -(2**16)) - self.starts[run]

            out[run] = time

        for ensemble_run, runs in self.ensembles.items():
            time = 0
            for run in runs:
                time += out.get(run, -(2**16))

            out[ensemble_run] = time

        return out

    def merge(self, tracker: "PerformanceTracker"):
        self._log_to_file |= tracker._log_to_file
        self.starts.update(tracker.starts)
        self.stops.update(tracker.stops)
        self.ensembles.update(tracker.ensembles)

    @staticmethod
    def get_trackers():
        return PerformanceTracker._trackers

    @staticmethod
    def merge_trackers(trackers: dict[str, "PerformanceTracker"]):
        for name, tracker in trackers.items():
            PerformanceTracker.get(name).merge(tracker)

    @staticmethod
    def is_multiprocess():
        from .progress import MULTIPROCESS_ENABLE

        return MULTIPROCESS_ENABLE

    @staticmethod
    def get(name: str):
        if name not in PerformanceTracker._trackers:
            nt = PerformanceTracker()
            PerformanceTracker._trackers[name] = nt
            return nt

        return PerformanceTracker._trackers[name]

    @staticmethod
    def log():
        import mlflow
        from .mlflow import mlflow_log_as_str

        if not mlflow.active_run():
            return

        multi = PerformanceTracker.is_multiprocess()
        mlflow.log_param("perf.multiprocess", str(multi))

        file_perfs = {}
        json_perfs = {}
        for tracker_name, tracker in PerformanceTracker.get_trackers().items():
            if not tracker._log_to_file:
                mlflow.log_param(f"perf.{tracker_name}.gpu", tracker.gpu)

            for name, metric in tracker.get_log_dict().items():
                metric_name = f"{tracker_name}.{name}"

                if metric < 0:
                    metric = -1
                    logger.warning(
                        f"Metric {metric_name} is negative, there is a missing `start()`, `stop()` or partial `ensemble()`."
                    )
                else:
                    # convert to seconds
                    metric /= 10**9

                ms = (metric % 1) * 1000
                seconds = metric % 60
                minutes = (metric // 60) % 60
                hours = metric // 3600
                str_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}:{int(ms):03d}"

                file_perfs[metric_name] = str_time
                json_perfs[metric_name] = metric
                if not tracker._log_to_file:
                    # mlflow.log_metric(f"perf.{metric_name}", metric)
                    mlflow.set_tag(
                        f"perf.{metric_name}",
                        str_time,
                    )

        file_txt = "\n".join(f"{k:65s} | {v}" for k, v in file_perfs.items())
        # FIXME: hardcoded var, should point to metrics dir
        mlflow.log_dict(json_perfs, "_raw/perf.json")
        mlflow_log_as_str("perf", file_txt)
