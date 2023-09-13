from kedro.pipeline import Pipeline as pipeline

from ...metric import (
    ColumnMetricFactory,
    MetricFactory,
    fit_column_holder,
    fit_metric,
    log_metric,
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


def _get_metric_encs(view: View, encodings: list[str] | str):
    if isinstance(encodings, str):
        return f"{view}.enc.{encodings}" if encodings != "raw" else {}
    else:
        return {
            enc: ({t: f"{view}.enc.{t}" for t in view.tables})
            for enc in encodings
            if enc != "raw"
        }


def _get_metric_data(view: View, split: str, encodings: list[str] | str):
    if isinstance(encodings, str):
        if encodings == "raw":
            return {t: f"{view}.{split}.{t}" for t in view.tables}
        else:
            return f"{view}.{split}.{encodings}"
    else:
        return {
            enc: (
                {t: f"{view}.{split}.{t}" for t in view.tables}
                if enc == "raw"
                else f"{view}.{split}.{enc}"
            )
            for enc in encodings
        }


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


def create_metrics_model_pipeline(
    view: View, alg: str, wrk_split: str, ref_split: str, modules: list[Module]
):
    nodes = []

    for fn in ("visualise", "summarize"):
        metrics = [*get_module_dict(MetricFactory, modules).keys(), "col"]
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
