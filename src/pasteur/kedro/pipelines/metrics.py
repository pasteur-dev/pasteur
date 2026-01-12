from typing import Any

from kedro.pipeline import Pipeline as pipeline

from ...metric import ColumnMetricFactory, MetricFactory, fit_column_holder, fit_metric
from ...module import Module, get_module_dict, get_module_dict_multiple
from ...utils import apply_fun
from ...utils.mlflow import mlflow_log_artifacts
from ...view import View
from .meta import TAGS_METRICS_INGEST, TAGS_METRICS_LOG
from .meta import DatasetMeta as D
from .meta import PipelineMeta, node


def _log_metajson(meta):
    return {"algorithm": meta.algs, "override": meta.alg_override}


def _log_metadata(view: View):
    return PipelineMeta(
        pipeline(
            [
                node(
                    func=mlflow_log_artifacts,
                    name=f"log_metadata",
                    inputs={"meta": f"{view}.metadata"},
                    outputs=None,
                    namespace=str(view),
                ),
                node(
                    func=_log_metajson,
                    name=f"log_metajson",
                    inputs={"meta": f"{view}.metadata"},
                    outputs=f"{view}.metajson",
                    namespace=str(view),
                ),
            ],
            tags=TAGS_METRICS_LOG,
        ),
        outputs=[
            D(
                "measure",
                f"{view}.metajson",
                ["view", view, "metajson"],
                type="json",
                versioned=True,
            ),
        ],
    )


def _get_metric_encs(view: View, encodings: list[str] | str):
    if isinstance(encodings, str):
        return f"{view}.enc.{encodings}" if encodings not in ("raw", "bst") else {}
    else:
        return {
            enc: f"{view}.enc.{enc}"
            for enc in encodings
            if enc not in ("raw", "bst")
        }


def _get_metric_data(view: View, split: str, encodings: list[str] | str):
    if isinstance(encodings, str):
        strip = True
        encodings = [encodings]
    else:
        strip = False

    out = {}
    for enc in encodings:
        if enc == "raw":
            out[enc] = {t: f"{view}.{split}.{t}" for t in view.tables}
        elif enc == "bst":
            out[enc] = {t: f"{view}.{split}.bst_{t}" for t in view.tables}
        else:
            out[enc] = f"{view}.{split}.{enc}"

    return next(iter(out.values())) if strip else out


def _create_fit_pipeline(
    view: View, modules: list[Module], fit_split: str, wrk_split: str, ref_split: str
):
    nodes = []
    outputs = []

    metric_factories = {**get_module_dict(MetricFactory, modules), "col": None}
    for name, fs in metric_factories.items():
        # Create node inputs
        if name == "col":
            enc = "raw"
            inputs_fit = {
                "metadata": f"{view}.metadata",
                "trns": {t: f"{view}.trn.{t}" for t in view.tables},
                "data": _get_metric_data(view, fit_split, "raw"),
            }
            nodes += [
                node(
                    fit_column_holder,
                    args=[modules],
                    name=f"fit_column_metrics",
                    inputs=inputs_fit,
                    outputs=f"{view}.msr.{name}",
                    namespace=f"{view}.msr",
                ),
            ]
        else:
            enc = fs.encodings
            inputs_fit = {
                "metadata": f"{view}.metadata",
                "encoder": _get_metric_encs(view, enc),
                "data": _get_metric_data(view, fit_split, enc),
            }
            nodes += [
                node(
                    fit_metric,
                    args=[fs],
                    name=f"fit_{name}",
                    inputs=inputs_fit,
                    outputs=f"{view}.msr.{name}",
                    namespace=f"{view}.msr",
                ),
            ]

        # Create node
        inputs_pre = {
            "obj": f"{view}.msr.{name}",
            "wrk": _get_metric_data(view, wrk_split, enc),
            "ref": _get_metric_data(view, ref_split, enc),
        }
        nodes += [
            node(
                name=f"preprocess_{name}",
                func=apply_fun,
                kwargs={"_fun": "preprocess"},
                inputs=inputs_pre,
                outputs=f"{view}.msr.{name}_pre",
                namespace=f"{view}.msr",
            ),
        ]
        outputs += [
            D(
                "measure",
                f"{view}.msr.{name}",
                ["view", view, "msr", name, "metric"],
                type="pkl",
            ),
            D(
                "measure",
                f"{view}.msr.{name}_pre",
                ["view", view, "msr", name, "pre"],
                type="pkl",
            ),
        ]
    return PipelineMeta(pipeline(nodes, tags=TAGS_METRICS_INGEST), outputs)


def _create_process_pipeline(
    view: View, modules: list[Module], syn_split: str, wrk_split: str, ref_split: str
):
    nodes = []
    outputs = []
    metric_factories = {**get_module_dict(MetricFactory, modules), "col": None}
    for name, fs in metric_factories.items():
        if name == "col":
            enc = "raw"
        else:
            enc = fs.encodings

        # Create node inputs
        inputs = {
            "obj": f"{view}.msr.{name}",
            "wrk": _get_metric_data(view, wrk_split, enc),
            "ref": _get_metric_data(view, ref_split, enc),
            "syn": _get_metric_data(view, syn_split, enc),
            "pre": f"{view}.msr.{name}_pre",
        }

        # Create node
        nodes += [
            node(
                name=f"process_{name}",
                func=apply_fun,
                kwargs={"_fun": "process"},
                inputs=inputs,
                outputs=f"{view}.{syn_split}.{name}_data",
                namespace=f"{view}.{syn_split}",
            ),
        ]
        outputs += [
            D(
                "measure",
                f"{view}.{syn_split}.{name}_data",
                ["synth", view, syn_split, "msr", name, "pre"],
                type="pkl",
                versioned=True,
            ),
        ]
    return PipelineMeta(pipeline(nodes, tags=TAGS_METRICS_LOG), outputs)


def create_metrics_ingest_pipeline(
    view: View, modules: list[Module], fit_split: str, wrk_split: str, ref_split: str
):
    return _log_metadata(view) + _create_fit_pipeline(
        view, modules, fit_split, wrk_split, ref_split
    )


def log_metric(metric: Any, data: Any):
    from ...utils.mlflow import mlflow_log_artifacts

    mlflow_log_artifacts("metrics", metric.unique_name(), metric=metric, data=data)


def log_model(model):
    from ...utils.mlflow import mlflow_log_as_str

    # TODO: Enable uploading model
    # mlflow_log_artifacts('model', model=model)
    mlflow_log_as_str("model", str(model))


def create_metrics_model_pipeline(
    view: View, alg: str, wrk_split: str, ref_split: str, modules: list[Module]
):
    metrics = [*get_module_dict(MetricFactory, modules).keys(), "col"]
    nodes = [
        node(
            func=log_model,
            name=f"upload_model",
            inputs=[f"{view}.{alg}.model"],
            outputs=None,
            namespace=f"{view}.{alg}",
        )
    ]

    for name in metrics:
        nodes += [
            node(
                func=log_metric,
                name=f"upload_{name}",
                inputs={
                    "metric": f"{view}.msr.{name}",
                    "data": f"{view}.{alg}.{name}_data",
                },
                outputs=None,
                namespace=f"{view}.{alg}",
            )
        ]

    for fn in ("visualise", "summarize"):
        for name in metrics:
            nodes += [
                node(
                    func=apply_fun,
                    name=f"{fn}_{name}",
                    kwargs={"_fun": fn},
                    inputs={
                        "obj": f"{view}.msr.{name}",
                        "data": {"syn": f"{view}.{alg}.{name}_data"},
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
    types.add("raw")

    for fs in get_module_dict(MetricFactory, modules).values():
        enc = fs.encodings
        if isinstance(enc, str):
            types.add(enc)
        else:
            types.update(enc)

    # Sort to ensure determinism
    return sorted(list(types))
