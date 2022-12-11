from kedro.pipeline import Pipeline as pipeline

from ...metric import (
    ColumnMetricFactory,
    DatasetMetricFactory,
    TableMetricFactory,
    fit_column_holder,
    fit_dataset_metric,
    fit_table_metric,
)
from ...module import Module, get_module_dict, get_module_dict_multiple
from ...utils.mlflow import mlflow_log_artifacts
from ...view import View
from ...utils import apply_fun
from .meta import DatasetMeta as D
from .meta import PipelineMeta, node
from .utils import gen_closure

from .meta import TAGS_METRICS_INGEST, TAGS_METRICS_LOG


def _log_metadata(view: View):
    return pipeline(
        [
            node(
                func=mlflow_log_artifacts,
                name=f"log_metadata",
                inputs={"meta": f"{view}.metadata"},
                outputs=None,
                namespace=str(view),
            )
        ],
        tags=TAGS_METRICS_INGEST,
    )


def _get_dataset_data(view: View, split: str, encodings: list[str]):
    data = {
        "ids": {t: f"{view}.{split}.ids_{t}" for t in view.tables},
        "tables": {},
    }
    for enc in encodings:
        if enc == "raw":
            data["tables"]["raw"] = {t: f"{view}.{split}.{t}" for t in view.tables}
        else:
            data["tables"][enc] = {t: f"{view}.{split}.{enc}_{t}" for t in view.tables}
    return data


def _get_table_data(view: View, split: str, table: str, encodings: list[str]):
    data = {
        "tables": {},
        "ids": f"{view}.{split}.ids_{table}",
    }
    input_tables = [*view.trn_deps.get(table, []), table]

    for enc in encodings:
        if enc == "raw":
            data["tables"]["raw"] = {t: f"{view}.{split}.{t}" for t in input_tables}
        else:
            data["tables"][enc] = {t: f"{view}.{split}.{enc}_{t}" for t in input_tables}

    return data


def _get_column_data(view: View, split: str, table: str):
    input_tables = [*view.trn_deps.get(table, []), table]
    return {
        "ids": f"{view}.{split}.ids_{table}",
        "tables": {t: f"{view}.{split}.{t}" for t in input_tables},
    }


def _create_fit_pipeline(
    view: View, modules: list[Module], fit_split: str, wrk_split: str, ref_split: str
):
    nodes = []
    outputs = []

    # Dataset Metrics
    for name, fs in get_module_dict(DatasetMetricFactory, modules).items():
        # Create node inputs
        inputs_fit = {
            "meta": f"{view}.metadata",
            "trns": {t: f"{view}.trn.{t}" for t in view.tables},
            "data": _get_dataset_data(view, fit_split, fs.encodings),
        }
        inputs_pre = {
            "obj": f"{view}.msr.ds_{name}",
            "wrk": _get_dataset_data(view, wrk_split, fs.encodings),
            "ref": _get_dataset_data(view, ref_split, fs.encodings),
        }

        # Create node
        nodes += [
            node(
                fit_dataset_metric,
                name=f"fit_{name}",
                inputs=inputs_fit,
                outputs=f"{view}.msr.ds_{name}",
                namespace=f"{view}.msr",
            ),
            node(
                name=f"preprocess_ds_{name}",
                func=apply_fun,
                kwargs={"_fun": "preprocess"},
                inputs=inputs_pre,
                outputs=f"{view}.msr.ds_{name}_pre",
                namespace=f"{view}.msr",
            ),
        ]
        outputs += {
            D(
                "measure",
                f"{view}.msr.ds_{name}",
                ["measure", "dataset", view, name, "metric"],
                type="pkl",
            ),
            D(
                "measure",
                f"{view}.msr.ds_{name}_pre",
                ["measure", "dataset", view, name, "pre"],
                type="pkl",
            ),
        }

    # Table Metrics
    for name, fs in get_module_dict(TableMetricFactory, modules).items():
        for table in view.tables:
            # Create node inputs
            input_tables = [*view.trn_deps.get(table, []), table]
            inputs_fit = {
                "meta": f"{view}.metadata",
                "trns": {t: f"{view}.trn.{t}" for t in input_tables},
                "data": _get_table_data(view, fit_split, table, fs.encodings),
            }
            inputs_pre = {
                "obj": f"{view}.msr.tbl_{name}_{table}",
                "wrk": _get_table_data(view, wrk_split, table, fs.encodings),
                "ref": _get_table_data(view, ref_split, table, fs.encodings),
            }

            # Create node
            nodes += [
                node(
                    gen_closure(fit_table_metric, fs, table, _fn=f"fit_{name}_{table}"),
                    inputs=inputs_fit,
                    outputs=f"{view}.msr.tbl_{name}_{table}",
                    namespace=f"{view}.msr",
                ),
                node(
                    name=f"preprocess_tbl_{name}_{table}",
                    func=apply_fun,
                    kwargs={"_fun": "preprocess"},
                    inputs=inputs_pre,
                    outputs=f"{view}.msr.tbl_{name}_{table}_pre",
                    namespace=f"{view}.msr",
                ),
            ]
            outputs += [
                D(
                    "measure",
                    f"{view}.msr.tbl_{name}_{table}",
                    ["measure", "table", view, table, name, "metric"],
                    type="pkl",
                ),
                D(
                    "measure",
                    f"{view}.msr.tbl_{name}_{table}_pre",
                    ["measure", "dataset", view, table, name, "pre"],
                    type="pkl",
                ),
            ]

    # Column Metrics
    col_modules = get_module_dict_multiple(ColumnMetricFactory, modules)
    if col_modules:
        for table in view.tables:
            # Create node inputs
            inputs_fit = {
                "meta": f"{view}.metadata",
                "data": _get_column_data(view, fit_split, table),
            }

            # Column metrics
            inputs_pre = {
                "obj": f"{view}.msr.col_{table}",
                "wrk": _get_column_data(view, wrk_split, table),
                "ref": _get_column_data(view, ref_split, table),
            }

            # Create node
            nodes += [
                node(
                    name=f"fit_column_metrics_{table}",
                    func=fit_column_holder,
                    args=[col_modules, table],
                    inputs=inputs_fit,
                    outputs=f"{view}.msr.col_{table}",
                    namespace=f"{view}.msr",
                ),
                node(
                    name=f"preprocess_col_metrics",
                    func=apply_fun,
                    kwargs={"_fun": "preprocess"},
                    inputs=inputs_pre,
                    outputs=f"{view}.msr.col_{table}_pre",
                    namespace=f"{view}.msr",
                ),
            ]
            outputs += [
                D(
                    "measure",
                    f"{view}.msr.col_{table}",
                    ["measure", "column", view, table, "metric"],
                    type="pkl",
                ),
                D(
                    "measure",
                    f"{view}.msr.col_{table}_pre",
                    ["measure", "dataset", view, table, "col", "pre"],
                    type="pkl",
                ),
            ]

    return PipelineMeta(pipeline(nodes, tags=TAGS_METRICS_INGEST), outputs)


def _create_process_pipeline(
    view: View, modules: list[Module], syn_split: str, wrk_split: str, ref_split: str
):
    nodes = []
    outputs = []

    # Dataset Metrics
    for name, fs in get_module_dict(DatasetMetricFactory, modules).items():
        # Create node inputs

        inputs = {
            "obj": f"{view}.msr.ds_{name}",
            "wrk": _get_dataset_data(view, wrk_split, fs.encodings),
            "ref": _get_dataset_data(view, ref_split, fs.encodings),
            "syn": _get_dataset_data(view, syn_split, fs.encodings),
        }

        # Create node
        nodes += [
            node(
                name=f"process_ds_{name}",
                func=apply_fun,
                kwargs={"_fun": "process"},
                inputs=inputs,
                outputs=f"{view}.{syn_split}.ds_{name}_data",
                namespace=f"{view}.{syn_split}",
            ),
        ]
        outputs += {
            D(
                "measure",
                f"{view}.msr.ds_{name}_data",
                ["synth", "measure", "dataset", view, name, "data"],
                type="pkl",
                versioned=True,
            ),
        }

    # Table Metrics
    for name, fs in get_module_dict(TableMetricFactory, modules).items():
        for table in view.tables:
            # Create node inputs

            inputs = {
                "obj": f"{view}.msr.tbl_{name}_{table}",
                "pre": f"{view}.msr.tbl_{name}_{table}_pre",
                "wrk": _get_table_data(view, wrk_split, table, fs.encodings),
                "ref": _get_table_data(view, ref_split, table, fs.encodings),
                "syn": _get_table_data(view, syn_split, table, fs.encodings),
            }

            # Create node
            nodes += [
                node(
                    name=f"process_tbl_{name}_{table}",
                    func=apply_fun,
                    kwargs={"_fun": "process"},
                    inputs=inputs,
                    outputs=f"{view}.{syn_split}.tbl_{name}_{table}_data",
                    namespace=f"{view}.{syn_split}",
                ),
            ]
            outputs += [
                D(
                    "measure",
                    f"{view}.{syn_split}.tbl_{name}_{table}_data",
                    ["synth", "measure", "table", view, table, name, "data"],
                    type="pkl",
                    versioned=True,
                ),
            ]

    # Column Metrics
    col_modules = get_module_dict_multiple(ColumnMetricFactory, modules)
    if col_modules:
        for table in view.tables:

            # Column metrics
            inputs = {
                "obj": f"{view}.msr.col_{table}",
                "pre": f"{view}.msr.col_{table}_pre",
                "wrk": _get_column_data(view, wrk_split, table),
                "ref": _get_column_data(view, ref_split, table),
                "syn": _get_column_data(view, syn_split, table),
            }

            # Create node
            nodes += [
                node(
                    name=f"process_col_{table}",
                    func=apply_fun,
                    kwargs={"_fun": "process"},
                    inputs=inputs,
                    outputs=f"{view}.{syn_split}.col_{table}_data",
                    namespace=f"{view}.{syn_split}",
                ),
            ]
            outputs += [
                D(
                    "measure",
                    f"{view}.{syn_split}.col_{table}_data",
                    ["synth", "measure", "column", view, table, "data"],
                    type="pkl",
                    versioned=True,
                ),
            ]

    return PipelineMeta(pipeline(nodes, tags=TAGS_METRICS_LOG), outputs)


def create_metrics_ingest_pipeline(
    view: View, modules: list[Module], wrk_split: str, ref_split: str
):
    return _log_metadata(view) + _create_fit_pipeline(
        view, modules, wrk_split, wrk_split, ref_split
    )


def create_metrics_model_pipeline(
    view: View, alg: str, wrk_split: str, ref_split: str, modules: list[Module]
):
    nodes = []

    for fn in ("visualise", "summarize"):
        col_modules = get_module_dict_multiple(ColumnMetricFactory, modules)
        if col_modules:
            for table in view.tables:
                nodes += [
                    node(
                        func=apply_fun,
                        name=f"{fn}_columns_{table}",
                        kwargs={"_fun": fn},
                        inputs={
                            "obj": f"{view}.msr.col_{table}",
                            "data": {"syn": f"{view}.{alg}.col_{table}_data"},
                        },
                        outputs=None,
                        namespace=f"{view}.{alg}",
                    )
                ]

        for name in get_module_dict(TableMetricFactory, modules):
            for table in view.tables:
                nodes += [
                    node(
                        func=apply_fun,
                        name=f"{fn}_{name}_{table}",
                        kwargs={"_fun": fn},
                        inputs={
                            "obj": f"{view}.msr.tbl_{name}_{table}",
                            "data": {"syn": f"{view}.{alg}.tbl_{name}_{table}_data"},
                        },
                        outputs=None,
                        namespace=f"{view}.{alg}",
                    )
                ]

        for name in get_module_dict(DatasetMetricFactory, modules):
            nodes += [
                node(
                    func=apply_fun,
                    name=f"{fn}_{name}",
                    kwargs={"_fun": fn},
                    inputs={
                        "obj": f"{view}.msr.ds_{name}",
                        "data": {"syn": f"{view}.{alg}.ds_{name}_data"},
                    },
                    outputs=None,
                    namespace=f"{view}.{alg}",
                )
            ]

    return _create_process_pipeline(
        view, modules, alg, wrk_split, ref_split
    ) + pipeline(nodes, tags=TAGS_METRICS_LOG)


def get_metrics_types(modules: list[Module]):
    types: set[str] = set()

    if len(get_module_dict_multiple(ColumnMetricFactory, modules)):
        types.add("raw")

    for fs in get_module_dict(DatasetMetricFactory, modules).values():
        types.update(fs.encodings)

    for fs in get_module_dict(TableMetricFactory, modules).values():
        types.update(fs.encodings)

    # Sort to ensure determinism
    return sorted(list(types))
