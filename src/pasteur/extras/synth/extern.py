from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ...synth import Synth

if TYPE_CHECKING:
    import pandas as pd

    from ...attribute import Attributes, IdxValue

logger = logging.getLogger(__name__)


class ExternalPythonSynth(Synth, ABC):
    """Abstract class for calling synthesis algorithms written in python
    from other projects.

    Can use a custom virtual environment, custom project directory, and custom command.
    With parameters defined per execution."""

    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    gpu = True

    _pg_args = {"index": False}

    def __init__(
        self,
        cmd: str,
        venv: str | None = None,
        dir: str | None = None,
        rebalance_columns: bool = True,
        **_,
    ) -> None:
        super().__init__(**_)
        self.venv = venv
        self.dir = dir
        self.cmd = cmd
        self.rebalance_columns = rebalance_columns

    @property
    def _logger(self):
        return logging.getLogger(f"extern.{self.name if self.name else 'ukn'}")

    def preprocess(
        self,
        attrs: dict[str, Attributes],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        self.attrs = attrs

    def bake(
        self,
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        pass

    def fit(self, data: dict[str, pd.DataFrame], ids: dict[str, pd.DataFrame]):
        import json
        import shlex
        import signal
        import subprocess
        from os import path
        from tempfile import TemporaryDirectory
        from threading import Thread

        import pandas as pd

        param_dict, data_in, data_out = self._prepare_synthesis_data(
            self.attrs, data, ids
        )

        with TemporaryDirectory(f"_{self.name}") as dir:
            fns = {}
            # Save input data to temporary files
            for name, (type, data) in data_in.items():
                match type:
                    case "json":
                        fn = path.join(dir, f"{name}.json")
                        with open(fn, "w") as f:
                            json.dump(data, f)
                    case "csv":
                        fn = path.join(dir, f"{name}.csv")
                        assert isinstance(data, pd.DataFrame)
                        data.to_csv(fn, **self._pg_args)
                    case _:
                        assert False, "Unknown type"
                fns[name] = fn

            for name, type in data_out.items():
                fns[name] = path.join(dir, f"{name}.{type}")

            # Create command
            # python3 or path/to/venv/bin/python3
            python_bin = path.join(self.venv, "bin/python3") if self.venv else "python3"

            # can call a module with -m, or include a dir and make the command a
            # python script
            if self.dir:
                # path/to/proj/test.py
                cmd = path.join(self.dir, self.cmd)
            else:
                # -m kedro ...
                cmd = self.cmd

            # Unfold the dictionary
            params = ""

            if isinstance(param_dict, dict):
                for param, val in param_dict.items():
                    params += f"{param} {val} "
            else:
                for param in param_dict:
                    params += f"{param} "

            full_cmd = f"{python_bin} {cmd} {params}"

            # Replace file names with directory
            full_cmd = full_cmd.format_map(fns)

            # Run command
            self._logger.info(f"Running command {self.cmd} as: \n{full_cmd}")

            def _log_pipe(proc, is_err):
                if is_err:
                    f = self._logger.warn
                    s = proc.stderr
                else:
                    f = self._logger.info
                    s = proc.stdout

                while True:
                    line = s.readline()
                    if not line:
                        break
                    f(line[:-1])

            proc = None
            tout = None
            terr = None
            try:

                def preexec_function():
                    signal.signal(signal.SIGINT, signal.SIG_IGN)

                proc = subprocess.Popen(
                    shlex.split(full_cmd),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=preexec_function,
                )

                tout = Thread(target=_log_pipe, args=(proc, False))
                terr = Thread(target=_log_pipe, args=(proc, True))
                tout.start()
                terr.start()

                proc.wait()

                if tout:
                    tout.join()
                if terr:
                    terr.join()
            except Exception:
                if proc and not proc.poll():
                    proc.terminate()
                if tout:
                    tout.join()
                if terr:
                    terr.join()
                raise

            # Load output data
            loaded_out = {}
            for name, type in data_out.items():
                fn = fns[name]
                match type:
                    case "csv":
                        loaded_out[name] = pd.read_csv(fn)
                    case "json":
                        loaded_out[name] = json.load(fn)

            self.loaded_out = loaded_out

    def sample(
        self, n: int | None = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        assert n is None, "Variable n not supported"

        return self._return_synthetic_data(self.loaded_out)

    @abstractmethod
    def _prepare_synthesis_data(
        self,
        attrs: dict[str, Attributes],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, Any] | list[Any], dict[str, tuple[str, Any]], dict[str, str]]:
        """Called during fit.

        Should return a parameter dictionary for the synthesis command,
        a dictionary of `<name>` to `(<type>,<data>)` tuples, where type is json or csv,
        and a dictionary of <name> to <type> tuples for output data

        The data in the input dictionary is saved to the filesystem in a temporary
        directory.

        The names in the output dictionary are associated with temporary files.

        Parameters containing `{name}` are substituted by the filename of
        the file with that name in the input/output dictionary.
        """
        pass

    @abstractmethod
    def _return_synthetic_data(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Receives as input a dictionary of `<name>` to `<data>` pairs.

        This function is expected to output a data, ids pair of DataFrame dictionaries,
        in the same format the sample function expects."""
        pass


class AimSynth(ExternalPythonSynth):
    name = "aim"
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    gpu = True

    def __init__(
        self,
        e: float = 1,
        delta: float | None = None,
        max_model_size: int = 80,
        rebalance_columns: bool = False,
        seed: int | None = None,
        deterministic_upsample: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(cmd="mechanisms/aim.py", **kwargs)
        self.e = e
        self.delta = delta
        self.max_model_size = max_model_size
        self.seed = seed
        self.rebalance_columns = rebalance_columns
        self.deterministic_upsample = deterministic_upsample
        self.kwargs = kwargs

    def _prepare_synthesis_data(
        self,
        attrs: dict[str, Attributes],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, Any] | list[Any], dict[str, tuple[str, Any]], dict[str, str]]:
        import pandas as pd

        assert len(data) == 1
        self.table_name = next(iter(data))
        table = data[self.table_name]
        table_attrs = attrs[self.table_name]

        if self.rebalance_columns:
            from ...hierarchy import rebalance_attributes

            table_attrs = rebalance_attributes(
                table, table_attrs, reshape_domain=False, **self.kwargs
            )
        self.table_attrs = table_attrs

        n = len(table)

        params = {
            "--dataset": "{dataset}",
            "--domain": "{domain}",
            "--save": "{save}",
            "--epsilon": self.e,
            "--delta": self.delta if self.delta != None else 1 / n / 10,
            "--max_model_size": self.max_model_size,
        }

        domain = {}
        downsampled_table = pd.DataFrame(index=table.index)
        for attr in table_attrs.values():
            for name, col in attr.vals.items():
                assert isinstance(col, IdxValue)
                h = col.select_height()
                domain[name] = col.get_domain(h)
                downsampled_table[name] = col.downsample(table[name].to_numpy(), h)

        data_in = {"dataset": ("csv", downsampled_table), "domain": ("json", domain)}
        data_out = {"save": "csv"}

        return params, data_in, data_out

    def _return_synthetic_data(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        import pandas as pd

        if self.rebalance_columns:
            downsampled_table = data["save"]
            table = pd.DataFrame(index=downsampled_table.index)

            for attr in self.table_attrs.values():
                for name, col in attr.vals.items():
                    assert isinstance(col, IdxValue)
                    h = col.select_height()
                    table[name] = col.upsample(
                        downsampled_table[name],
                        h,
                        deterministic=self.deterministic_upsample,
                    )
        else:
            table = data["save"]

        return {self.table_name: table}, {self.table_name: pd.DataFrame()}


class PrivMrfSynth(ExternalPythonSynth):
    """ Runs the PrivMrf algorithm externally.
    
    Place the following snippet in `<priv-mrf>/pasteur.py`:
    ```
    from sys import argv
    import PrivMRF
    import PrivMRF.utils.tools as tools
    from PrivMRF.domain import Domain
    import numpy as np

    if __name__ == '__main__':
        fn_data, fn_domain, fn_synth, e = argv
        e = float(e)

        data, _ = tools.read_csv(fn_data)
        data = np.array(data, dtype=int)

        json_domain = tools.read_json_domain(fn_domain)
        domain = Domain(json_domain, list(range(data.shape[1])))

        model = PrivMRF.run(data, domain, attr_hierarchy=None, \
            exp_name='exp', epsilon=e, p_config={})

        syn_data = model.synthetic_data(fn_synth)
    ```
    """

    name = "mrf"
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    gpu = True

    def __init__(
        self,
        e: float = 1.12,
        seed: int | None = None,
        rebalance_columns: bool | None = None,
        deterministic_upsample: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(cmd="pasteur.py", **kwargs)
        self.e = e
        self.seed = seed
        self.rebalance_columns = rebalance_columns
        self.deterministic_upsample = deterministic_upsample
        self.kwargs = kwargs

    def _prepare_synthesis_data(
        self,
        attrs: dict[str, Attributes],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        import pandas as pd

        assert len(data) == 1
        self.table_name = next(iter(data))
        table = data[self.table_name]
        table_attrs = attrs[self.table_name]

        params = ["{dataset}", "{domain}", "{out}", f"{self.e}"]

        if self.rebalance_columns:
            from ...hierarchy import rebalance_attributes

            table_attrs = rebalance_attributes(
                table, table_attrs, reshape_domain=False, **self.kwargs
            )

        col_names = []
        domain: dict[str, dict[str, Any]] = {}
        downsampled_table = pd.DataFrame(index=table.index)
        for attr in table_attrs.values():
            for name, col in attr.vals.items():
                assert isinstance(col, IdxValue)
                h = col.select_height()
                domain[str(len(col_names))] = {
                    "type": "discrete",
                    "domain": col.get_domain(h),
                }
                col_names.append(name)
                downsampled_table[name] = col.downsample(table[name].to_numpy(), h)

        self.id = table.index.name
        self.col_names = col_names
        self.table_attrs = table_attrs

        col_mapping = {name: i for i, name in enumerate(col_names)}
        dataset = downsampled_table.rename(columns=col_mapping)

        data_in = {"dataset": ("csv", dataset), "domain": ("json", domain)}
        data_out = {"out": "csv"}

        return params, data_in, data_out

    def _return_synthetic_data(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        import pandas as pd

        col_mapping = {str(i): name for i, name in enumerate(self.col_names)}
        downsampled_table = data["out"].rename(columns=col_mapping)

        if self.rebalance_columns:
            table = pd.DataFrame(index=downsampled_table.index)
            for attr in self.table_attrs.values():
                for name, col in attr.vals.items():
                    assert isinstance(col, IdxValue)
                    h = col.select_height()
                    table[name] = col.upsample(
                        downsampled_table[name],
                        h,
                        deterministic=self.deterministic_upsample,
                    )
        else:
            table = downsampled_table

        table.index.name = self.id
        return {self.table_name: table}, {self.table_name: pd.DataFrame()}
