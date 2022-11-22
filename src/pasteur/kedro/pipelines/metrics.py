from kedro.pipeline import Pipeline as pipeline
from kedro.pipeline import node

from ...metric import (
    ColumnMetricFactory,
    DatasetMetricFactory,
    Metric,
    TableMetricFactory,
    fit_column_holder,
    fit_dataset_metric,
    fit_table_metric,
    log_metric,
    process_column_holder,
    process_dataset_metric,
    process_table_metric,
    sum_metric,
    viz_metric,
)
from ...module import Module, get_module_dict, get_module_dict_multiple
from ...utils.mlflow import mlflow_log_artifacts
from ...view import View
from .meta import DatasetMeta as D
from .meta import PipelineMeta
from .utils import gen_closure

from .meta import TAGS_METRICS_INGEST, TAGS_METRICS_LOG

def _log_metadata(view: View):
    return pipeline(
        [
            node(
                gen_closure(mlflow_log_artifacts, _fn=f"log_metadata"),
                inputs={"meta": f"{view}.metadata"},
                outputs=None,
                namespace=str(view),
            )
        ], tags=TAGS_METRICS_INGEST
    )


def _create_fit_pipeline(view: View, modules: list[Module], split: str):
    nodes = []
    outputs = []

    # Dataset Metrics
    for name, fs in get_module_dict(DatasetMetricFactory, modules).items():
        # Create node inputs
        inputs = {"meta": f"{view}.metadata"}
        inputs.update({f"ids.{t}": f"{split}.{split}.ids_{t}" for t in view.tables})

        requires_attr = False
        for enc in fs.encodings:
            if enc == "raw":
                inputs.update({f"raw.{t}": f"{view}.{split}.{t}" for t in view.tables})
            else:
                requires_attr = True
                inputs.update(
                    {f"{enc}.{t}": f"{view}.{split}.{enc}_{t}" for t in view.tables}
                )

        if requires_attr:
            inputs.update({f"trn.{t}": f"{view}.trn.{t}" for t in view.tables})

        # Create node
        nodes += [
            node(
                gen_closure(fit_dataset_metric, _fn=f"fit_{name}"),
                inputs=inputs,
                outputs=f"{view}.msr.ds_{name}",
                namespace=f"{view}.msr",
            ),
        ]
        outputs += {
            D(
                "measure",
                f"{view}.msr.ds_{name}",
                ["measure", "dataset", view, name, "metric"],
                type="pkl",
            )
        }

    # Table Metrics
    for name, fs in get_module_dict(TableMetricFactory, modules).items():
        for table in view.tables:
            # Create node inputs
            input_tables = [*view.trn_deps.get(table, []), table]
            inputs = {"meta": f"{view}.metadata", "ids": f"{view}.{split}.ids_{table}"}

            requires_attr = False
            for enc in fs.encodings:
                if enc == "raw":
                    inputs.update(
                        {f"raw.{t}": f"{view}.{split}.{t}" for t in input_tables}
                    )
                else:
                    requires_attr = True
                    inputs.update(
                        {
                            f"{enc}.{t}": f"{view}.{split}.{enc}_{t}"
                            for t in input_tables
                        }
                    )

            if requires_attr:
                inputs.update({f"trn.{t}": f"{view}.trn.{t}" for t in input_tables})

            # Create node
            nodes += [
                node(
                    gen_closure(fit_table_metric, fs, table, _fn=f"fit_{name}_{table}"),
                    inputs=inputs,
                    outputs=f"{view}.msr.tbl_{name}_{table}",
                    namespace=f"{view}.msr",
                )
            ]
            outputs += [
                D(
                    "measure",
                    f"{view}.msr.tbl_{name}_{table}",
                    ["measure", "table", view, table, name, "metric"],
                    type="pkl",
                )
            ]

    # Column Metrics
    col_modules = get_module_dict_multiple(ColumnMetricFactory, modules)
    if col_modules:
        for table in view.tables:
            # Create node inputs
            input_tables = [*view.trn_deps.get(table, []), table]
            inputs = {
                "meta": f"{view}.metadata",
                "ids": f"{view}.{split}.ids_{table}",
                **{f"raw.{t}": f"{view}.{split}.{t}" for t in input_tables},
            }

            # Create node
            nodes += [
                node(
                    gen_closure(
                        fit_column_holder,
                        col_modules,
                        table,
                        _fn=f"fit_column_metrics_{table}",
                    ),
                    inputs=inputs,
                    outputs=f"{view}.msr.col_{table}",
                    namespace=f"{view}.msr",
                )
            ]
            outputs += [
                D(
                    "measure",
                    f"{view}.msr.col_{table}",
                    ["measure", "column", view, table, "metric"],
                    type="pkl",
                )
            ]

    return PipelineMeta(pipeline(nodes, tags=TAGS_METRICS_INGEST), outputs)


def _create_process_pipeline(
    view: View, modules: list[Module], split: str, split_type: int = 1
):
    nodes = []
    outputs = []

    if split_type == Metric.SYN_SPLIT:
        versioned = True
        pkg = split
        suffix = "data"
        out_dir = ["synth", "measure"]
        tags = TAGS_METRICS_LOG
    else:
        versioned = False
        pkg = "msr"
        suffix = split
        out_dir = ["measure"]
        tags = TAGS_METRICS_INGEST

    # Dataset Metrics
    for name, fs in get_module_dict(DatasetMetricFactory, modules).items():
        # Create node inputs
        inputs = {
            "metric": f"{view}.msr.ds_{name}",
            **{f"ids.{t}": f"{view}.{split}.ids_{t}" for t in view.tables},
        }
        for enc in fs.encodings:
            if enc == "raw":
                inputs.update({f"raw.{t}": f"{view}.{split}.{t}" for t in view.tables})
            else:
                inputs.update(
                    {f"{enc}.{t}": f"{view}.{split}.{enc}_{t}" for t in view.tables}
                )

        # Create node
        nodes += [
            node(
                gen_closure(
                    process_dataset_metric,
                    split=split_type,
                    _fn=f"process_{name}_{split}",
                ),
                inputs=inputs,
                outputs=f"{view}.{pkg}.ds_{name}_{suffix}",
                namespace=f"{view}.msr",
            ),
        ]
        outputs += {
            D(
                "measure",
                f"{view}.{pkg}.ds_{name}_{suffix}",
                [*out_dir, "dataset", view, name, suffix],
                type="pkl",
                versioned=versioned,
            )
        }

    # Table Metrics
    for name, fs in get_module_dict(TableMetricFactory, modules).items():
        for table in view.tables:
            # Create node inputs
            input_tables = [*view.trn_deps.get(table, []), table]
            inputs = {
                "metric": f"{view}.msr.tbl_{name}_{table}",
                "ids": f"{view}.{split}.ids_{table}",
            }

            for enc in fs.encodings:
                if enc == "raw":
                    inputs.update(
                        {f"raw.{t}": f"{view}.{split}.{t}" for t in input_tables}
                    )
                else:
                    inputs.update(
                        {
                            f"{enc}.{t}": f"{view}.{split}.{enc}_{t}"
                            for t in input_tables
                        }
                    )

            # Create node
            nodes += [
                node(
                    gen_closure(
                        process_table_metric,
                        split=split_type,
                        _fn=f"process_{name}_{table}_{split}",
                    ),
                    inputs=inputs,
                    outputs=f"{view}.{pkg}.tbl_{name}_{table}_{suffix}",
                    namespace=f"{view}.{pkg}",
                )
            ]
            outputs += [
                D(
                    "measure",
                    f"{view}.{pkg}.tbl_{name}_{table}_{suffix}",
                    [*out_dir, "table", view, table, name, suffix],
                    type="pkl",
                    versioned=versioned,
                )
            ]

    # Column Metrics
    col_modules = get_module_dict_multiple(ColumnMetricFactory, modules)
    if col_modules:
        for table in view.tables:
            # Create node inputs
            input_tables = [*view.trn_deps.get(table, []), table]
            inputs = {
                "holder": f"{view}.msr.col_{table}",
                "ids": f"{view}.{split}.ids_{table}",
                **{f"raw.{t}": f"{view}.{split}.{t}" for t in input_tables},
            }

            # Create node
            nodes += [
                node(
                    gen_closure(
                        process_column_holder,
                        split=split_type,
                        _fn=f"process_column_metrics_{table}_{split}",
                    ),
                    inputs=inputs,
                    outputs=f"{view}.{pkg}.col_{table}_{suffix}",
                    namespace=f"{view}.{pkg}",
                )
            ]
            outputs += [
                D(
                    "measure",
                    f"{view}.{pkg}.col_{table}_{suffix}",
                    [*out_dir, "column", view, table, suffix],
                    type="pkl",
                    versioned=versioned,
                )
            ]

    return PipelineMeta(pipeline(nodes, tags=tags), outputs)


def create_metrics_ingest_pipeline(
    view: View, modules: list[Module], wrk_split: str, ref_split: str
):
    return (
        _log_metadata(view)
        + _create_fit_pipeline(view, modules, wrk_split)
        + _create_process_pipeline(view, modules, wrk_split, Metric.WRK_SPLIT)
        + _create_process_pipeline(view, modules, ref_split, Metric.REF_SPLIT)
    )


def create_metrics_model_pipeline(
    view: View, alg: str, wrk_split: str, ref_split: str, modules: list[Module]
):
    nodes = []

    for fn in (viz_metric, sum_metric, log_metric):
        col_modules = get_module_dict_multiple(ColumnMetricFactory, modules)
        if col_modules:
            for table in view.tables:
                nodes += [
                    node(
                        gen_closure(fn, _fn=f"%s_columns_{table}"),
                        inputs={
                            "metric": f"{view}.msr.col_{table}",
                            "wrk": f"{view}.msr.col_{table}_{wrk_split}",
                            "ref": f"{view}.msr.col_{table}_{ref_split}",
                            "syn": f"{view}.{alg}.col_{table}_data",
                        },
                        outputs=None,
                        namespace=f"{view}.{alg}",
                    )
                ]

        for name in get_module_dict(TableMetricFactory, modules):
            for table in view.tables:
                nodes += [
                    node(
                        gen_closure(fn, _fn=f"%s_{name}_{table}"),
                        inputs={
                            "metric": f"{view}.msr.tbl_{name}_{table}",
                            "wrk": f"{view}.msr.tbl_{name}_{table}_{wrk_split}",
                            "ref": f"{view}.msr.tbl_{name}_{table}_{ref_split}",
                            "syn": f"{view}.{alg}.tbl_{name}_{table}_data",
                        },
                        outputs=None,
                        namespace=f"{view}.{alg}",
                    )
                ]

        for name in get_module_dict(DatasetMetricFactory, modules):
            nodes += [
                node(
                    gen_closure(fn, _fn=f"%s_{name}"),
                    inputs={
                        "metric": f"{view}.msr.ds_{name}",
                        "wrk": f"{view}.msr.ds_{name}_{wrk_split}",
                        "ref": f"{view}.msr.ds_{name}_{ref_split}",
                        "syn": f"{view}.{alg}.ds_{name}_data",
                    },
                    outputs=None,
                    namespace=f"{view}.{alg}",
                )
            ]

    return PipelineMeta(pipeline(nodes, tags=TAGS_METRICS_LOG), []) + _create_process_pipeline(
        view, modules, alg, Metric.SYN_SPLIT
    )


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
