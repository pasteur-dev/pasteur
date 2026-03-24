"""Flask application factory for LITMUS."""

import logging
import uuid
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, request

from .store import (
    Experiment,
    ExperimentStore,
    ModelRef,
    Rating,
    Run,
)

logger = logging.getLogger(__name__)


def create_app(
    data_dir: str | Path,
    catalog_info: dict | None = None,
    generator=None,
) -> Flask:
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="")

    store = ExperimentStore(data_dir)
    app.config["store"] = store
    app.config["catalog_info"] = catalog_info or {}
    app.config["generator"] = generator

    _register_routes(app)
    return app


def _get_store(app: Flask) -> ExperimentStore:
    return app.config["store"]


def _register_routes(app: Flask):
    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    # --- Views & Models ---

    @app.route("/api/views")
    def list_views():
        info = app.config["catalog_info"]
        views = list(info.get("views", {}).keys())
        return jsonify(views)

    @app.route("/api/views/<view>/models")
    def list_models(view: str):
        info = app.config["catalog_info"]
        models = info.get("views", {}).get(view, {}).get("models", {})
        return jsonify(models)

    # --- Experiments ---

    @app.route("/api/experiments", methods=["GET"])
    def list_experiments():
        store = _get_store(app)
        return jsonify([_exp_summary(e) for e in store.list_experiments()])

    @app.route("/api/experiments", methods=["POST"])
    def create_experiment():
        store = _get_store(app)
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400

        models = [ModelRef(**m) for m in data.get("models", [])]
        if not models:
            return jsonify({"error": "At least one model required"}), 400

        exp = store.create_experiment(
            name=data.get("name", ""),
            view=data["view"],
            models=models,
            include_real=data.get("include_real", True),
            blind=data.get("blind", True),
            samples_per_split=data.get("samples_per_split", 20),
        )
        return jsonify(_exp_detail(exp)), 201

    @app.route("/api/experiments/<eid>")
    def get_experiment(eid: str):
        store = _get_store(app)
        exp = store.get_experiment(eid)
        if not exp:
            return jsonify({"error": "Experiment not found"}), 404
        return jsonify(_exp_detail(exp))

    @app.route("/api/experiments/<eid>", methods=["DELETE"])
    def delete_experiment(eid: str):
        store = _get_store(app)
        if store.delete_experiment(eid):
            return jsonify({"ok": True})
        return jsonify({"error": "Experiment not found"}), 404

    # --- Runs ---

    def _preload_experiment_data(exp):
        """Preload entity data for all models in an experiment (background)."""
        generator = app.config.get("generator")
        if not generator:
            return
        import threading

        def _load():
            for m in exp.models:
                try:
                    generator._load_synth_data(exp.view, m.algorithm, m.timestamp)
                except Exception:
                    pass
            if exp.include_real:
                try:
                    generator._load_real_data(exp.view)
                except Exception:
                    pass
            logger.info(f"Preloaded data for experiment {exp.id}")

        threading.Thread(target=_load, daemon=True).start()

    @app.route("/api/experiments/<eid>/runs", methods=["POST"])
    def create_run(eid: str):
        store = _get_store(app)
        data = request.json or {}
        run = store.create_run(
            eid,
            name=data.get("name", ""),
            tutorial=data.get("tutorial", False),
        )
        if not run:
            return jsonify({"error": "Experiment not found"}), 404
        exp = store.get_experiment(eid)
        _preload_experiment_data(exp)
        return jsonify(_run_detail(run, exp)), 201

    @app.route("/api/experiments/<eid>/runs/<rid>")
    def get_run(eid: str, rid: str):
        store = _get_store(app)
        result = store.get_run(eid, rid)
        if not result:
            return jsonify({"error": "Run not found"}), 404
        exp, run = result
        return jsonify(_run_detail(run, exp))

    @app.route("/api/experiments/<eid>/runs/<rid>", methods=["DELETE"])
    def delete_run(eid: str, rid: str):
        store = _get_store(app)
        if store.delete_run(eid, rid):
            return jsonify({"ok": True})
        return jsonify({"error": "Run not found"}), 404

    @app.route("/api/experiments/<eid>/runs/<rid>/rate", methods=["POST"])
    def rate_entity(eid: str, rid: str):
        store = _get_store(app)
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400

        rating = Rating(
            entity_id=data["entity_id"],
            source=data["source"],
            score=data.get("score"),
            response_time_ms=data.get("response_time_ms"),
            skipped=data.get("skipped", False),
        )
        if store.add_rating(eid, rid, rating):
            result = store.get_run(eid, rid)
            exp, run = result if result else (None, None)
            return jsonify({
                "ok": True,
                "progress": run.progress if run else 0,
                "total": exp.total_samples if exp else 0,
            })
        return jsonify({"error": "Cannot add rating"}), 400

    @app.route("/api/experiments/<eid>/runs/<rid>/skip", methods=["POST"])
    def skip_entity(eid: str, rid: str):
        store = _get_store(app)
        data = request.json or {}
        rating = Rating(
            entity_id=data.get("entity_id", ""),
            source=data.get("source", ""),
            score=None,
            response_time_ms=None,
            skipped=True,
        )
        if store.add_rating(eid, rid, rating):
            return jsonify({"ok": True})
        return jsonify({"error": "Cannot skip"}), 400

    @app.route("/api/experiments/<eid>/runs/<rid>/undo", methods=["POST"])
    def undo_rating(eid: str, rid: str):
        store = _get_store(app)
        rating = store.undo_rating(eid, rid)
        if rating:
            result = store.get_run(eid, rid)
            progress = result[1].progress if result else 0
            return jsonify({"ok": True, "progress": progress})
        return jsonify({"error": "Nothing to undo"}), 400

    @app.route("/api/experiments/<eid>/runs/<rid>/end", methods=["POST"])
    def end_run(eid: str, rid: str):
        store = _get_store(app)
        if store.finish_run(eid, rid):
            return jsonify({"ok": True})
        return jsonify({"error": "Run not found"}), 404

    @app.route("/api/experiments/<eid>/runs/<rid>/resume", methods=["POST"])
    def resume_run(eid: str, rid: str):
        store = _get_store(app)
        if store.resume_run(eid, rid):
            return jsonify({"ok": True})
        return jsonify({"error": "Run not found"}), 404

    @app.route("/api/experiments/<eid>/results")
    def get_experiment_results(eid: str):
        store = _get_store(app)
        generator = app.config.get("generator")
        exp = store.get_experiment(eid)
        if not exp:
            return jsonify({"error": "Experiment not found"}), 404
        return jsonify(_compute_results(exp, generator))

    @app.route("/api/experiments/<eid>/runs/<rid>/results")
    def get_run_results(eid: str, rid: str):
        store = _get_store(app)
        result = store.get_run(eid, rid)
        if not result:
            return jsonify({"error": "Run not found"}), 404
        exp, run = result
        return jsonify(_compute_run_results(exp, run))

    # --- Entity Generation ---

    def _pick_source(exp, run):
        splits = []
        for m in exp.models:
            splits.append(("model", m.algorithm, m.timestamp))
        if exp.include_real:
            splits.append(("real", None, None))
        if not splits:
            return None, None, None
        split_idx = run.progress % len(splits)
        return splits[split_idx]

    def _generate_entity(generator, exp, source_type, alg, version):
        pretty_names = _compute_pretty_names(exp)
        if source_type == "real":
            entity = generator.generate_entity_from_real(exp.view)
            source = "real"
            source_pretty = pretty_names.get("real", "Real Data")
        else:
            entity = generator.generate_entity_from_model(
                exp.view, alg, version
            )
            source = f"{alg}_{version or 'latest'}"
            source_pretty = pretty_names.get(source, alg or "")
        return entity, source, source_pretty

    @app.route("/api/experiments/<eid>/runs/<rid>/next")
    def next_entity(eid: str, rid: str):
        store = _get_store(app)
        generator = app.config.get("generator")
        result = store.get_run(eid, rid)

        if not result:
            return jsonify({"error": "Run not found"}), 404
        exp, run = result
        if not run.started:
            return jsonify({"error": "Run not started"}), 400
        if run.finished:
            return jsonify({"error": "Run already finished"}), 400
        if not generator:
            return jsonify({"error": "No generator available"}), 500

        source_type, alg, version = _pick_source(exp, run)
        if source_type is None:
            return jsonify({"error": "No splits configured"}), 400

        entity_id = str(uuid.uuid4())
        try:
            entity, source, source_pretty = _generate_entity(
                generator, exp, source_type, alg, version
            )
        except Exception:
            logger.exception("Entity generation failed")
            return jsonify({"error": "Entity generation failed"}), 500

        return jsonify({
            "entity_id": entity_id,
            "source": source,
            "source_pretty": source_pretty,
            "entity": entity,
        })


# --- Serialization helpers ---

def _exp_summary(exp: Experiment) -> dict:
    pretty_names = _compute_pretty_names(exp)
    return {
        "id": exp.id,
        "name": exp.name,
        "view": exp.view,
        "models": [asdict(m) for m in exp.models],
        "num_models": len(exp.models),
        "include_real": exp.include_real,
        "blind": exp.blind,
        "samples_per_split": exp.samples_per_split,
        "total_samples": exp.total_samples,
        "num_runs": len(exp.runs),
        "created_at": exp.created_at,
        "pretty_names": pretty_names,
    }


def _exp_detail(exp: Experiment) -> dict:
    return {
        **_exp_summary(exp),
        "models": [asdict(m) for m in exp.models],
        "runs": [_run_summary(r, exp) for r in exp.runs],
    }


def _run_summary(run: Run, exp: Experiment) -> dict:
    return {
        "id": run.id,
        "name": run.name,
        "tutorial": run.tutorial,
        "started": run.started,
        "finished": run.finished,
        "progress": run.progress,
        "total_samples": exp.total_samples,
        "created_at": run.created_at,
    }


def _run_detail(run: Run, exp: Experiment | None) -> dict:
    base = _run_summary(run, exp) if exp else {"id": run.id}
    return {
        **base,
        "ratings": [asdict(r) for r in run.ratings],
    }


def _compute_pretty_names(exp: Experiment) -> dict[str, str]:
    """Compute display names for all sources in an experiment.

    Only shows override params that differ between selected models
    of the same algorithm. If only one version of an algorithm is
    selected, just shows the algorithm name.
    """
    from collections import defaultdict

    # Group models by algorithm
    by_alg: dict[str, list[ModelRef]] = defaultdict(list)
    for m in exp.models:
        by_alg[m.algorithm].append(m)

    names = {}
    for alg, models in by_alg.items():
        if len(models) == 1:
            # Single model of this algorithm — just the name
            key = f"{alg}_{models[0].timestamp or 'latest'}"
            names[key] = alg.title()
        else:
            # Multiple models — find differing params
            ref = models[0].overrides
            differing = set()
            for m in models[1:]:
                for k in set(list(ref.keys()) + list(m.overrides.keys())):
                    if ref.get(k) != m.overrides.get(k):
                        differing.add(k)

            for m in models:
                key = f"{alg}_{m.timestamp or 'latest'}"
                parts = [alg.title()]
                for k in sorted(differing):
                    if k in m.overrides:
                        parts.append(f"{k}={m.overrides[k]}")
                names[key] = " ".join(parts)

    names["real"] = "Real Data"
    return names


def _dist_entry(pretty_name: str, dist: dict[int, int]) -> dict:
    total = sum(dist.values())
    mean = sum(k * v for k, v in dist.items()) / total if total else 0
    return {
        "pretty_name": pretty_name,
        "count": total,
        "mean": mean,
        "distribution": dist,
    }


def _compute_results(exp: Experiment, generator=None) -> dict:
    from collections import defaultdict

    pretty_names = _compute_pretty_names(exp)

    # Build ordered source keys: models first, then real
    source_order = []
    for m in exp.models:
        source_order.append(f"{m.algorithm}_{m.timestamp or 'latest'}")
    if exp.include_real:
        source_order.append("real")

    # Collect human ratings
    by_source: dict[str, list[int]] = defaultdict(list)
    for run in exp.runs:
        if run.tutorial:
            continue
        for r in run.ratings:
            if not r.skipped and r.score is not None:
                by_source[r.source].append(r.score)

    # Build human results in source order
    human_results = []
    for source in source_order:
        scores = by_source.get(source, [])
        if not scores:
            continue
        dist = {i: scores.count(i) for i in range(1, 6)}
        human_results.append({
            "source": source,
            **_dist_entry(pretty_names.get(source) or source, dist),
        })

    # Load LLM evaluator scores in same source order
    llm_results = []
    llm_ref_dist = None
    if generator:
        for m in exp.models:
            source_key = f"{m.algorithm}_{m.timestamp or 'latest'}"
            try:
                syn_dist, ref_dist = generator.load_llm_scores(
                    exp.view, m.algorithm, m.timestamp
                )
                if syn_dist:
                    llm_results.append({
                        "source": source_key,
                        **_dist_entry(
                            pretty_names.get(source_key, source_key), syn_dist
                        ),
                    })
                if ref_dist and llm_ref_dist is None:
                    llm_ref_dist = ref_dist
            except Exception:
                pass

        if exp.include_real and llm_ref_dist:
            llm_results.append({
                "source": "real",
                **_dist_entry(pretty_names.get("real", "Real Data"), llm_ref_dist),
            })

    total_rated = sum(
        1 for run in exp.runs if not run.tutorial
        for r in run.ratings if not r.skipped
    )
    total_skipped = sum(
        1 for run in exp.runs if not run.tutorial
        for r in run.ratings if r.skipped
    )

    return {
        "experiment_id": exp.id,
        "name": exp.name,
        "view": exp.view,
        "num_runs": len([r for r in exp.runs if not r.tutorial]),
        "total_rated": total_rated,
        "total_skipped": total_skipped,
        "by_source": human_results,
        "llm_scores": llm_results,
    }


def _compute_run_results(exp: Experiment, run: Run) -> dict:
    from collections import defaultdict

    pretty_names = _compute_pretty_names(exp)

    source_order = []
    for m in exp.models:
        source_order.append(f"{m.algorithm}_{m.timestamp or 'latest'}")
    if exp.include_real:
        source_order.append("real")

    by_source: dict[str, list[int]] = defaultdict(list)
    for r in run.ratings:
        if not r.skipped and r.score is not None:
            by_source[r.source].append(r.score)

    results = []
    for source in source_order:
        scores = by_source.get(source, [])
        if not scores:
            continue
        dist = {i: scores.count(i) for i in range(1, 6)}
        results.append({
            "source": source,
            **_dist_entry(pretty_names.get(source) or source, dist),
        })

    total_rated = sum(1 for r in run.ratings if not r.skipped)
    total_skipped = sum(1 for r in run.ratings if r.skipped)

    return {
        "run_id": run.id,
        "name": run.name,
        "total_rated": total_rated,
        "total_skipped": total_skipped,
        "by_source": results,
    }
