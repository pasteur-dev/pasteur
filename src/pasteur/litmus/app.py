"""Flask application factory for LITMUS."""

import json
import logging
import random
import uuid
from pathlib import Path

from flask import Flask, jsonify, request

from .store import (
    Experiment,
    ExperimentStore,
    ModelRef,
    Rating,
    TimingParams,
)

logger = logging.getLogger(__name__)


def create_app(
    data_dir: str | Path,
    catalog_info: dict | None = None,
    generator=None,
) -> Flask:
    """Create the LITMUS Flask application.

    Args:
        data_dir: Path to Kedro data directory for experiment persistence.
        catalog_info: Dict with available views and models from filesystem scan.
        generator: EntityGenerator instance for generating entities.
    """
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")

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
        experiments = store.list_all()
        return jsonify([_experiment_summary(e) for e in experiments])

    @app.route("/api/experiments", methods=["POST"])
    def create_experiment():
        store = _get_store(app)
        data = request.json
        if not data:
            return jsonify({"error": "Request body required"}), 400

        models = [ModelRef(**m) for m in data.get("models", [])]
        if not models:
            return jsonify({"error": "At least one model required"}), 400

        exp = store.create(
            view=data["view"],
            models=models,
            include_real=data.get("include_real", True),
            blind=data.get("blind", True),
            samples_per_split=data.get("samples_per_split", 20),
        )
        return jsonify(_experiment_detail(exp)), 201

    @app.route("/api/experiments/<experiment_id>")
    def get_experiment(experiment_id: str):
        store = _get_store(app)
        exp = store.get(experiment_id)
        if not exp:
            return jsonify({"error": "Experiment not found"}), 404
        return jsonify(_experiment_detail(exp))

    @app.route("/api/experiments/<experiment_id>", methods=["DELETE"])
    def delete_experiment(experiment_id: str):
        store = _get_store(app)
        if store.delete(experiment_id):
            return jsonify({"ok": True})
        return jsonify({"error": "Experiment not found"}), 404

    @app.route("/api/experiments/<experiment_id>/start", methods=["POST"])
    def start_experiment(experiment_id: str):
        store = _get_store(app)
        data = request.json or {}
        name = data.get("name", "")
        tutorial = data.get("tutorial", False)
        if store.start_experiment(experiment_id, name, tutorial):
            return jsonify({"ok": True})
        return jsonify({"error": "Experiment not found"}), 404

    @app.route("/api/experiments/<experiment_id>/rate", methods=["POST"])
    def rate_entity(experiment_id: str):
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
        if store.add_rating(experiment_id, rating):
            exp = store.get(experiment_id)
            return jsonify({
                "ok": True,
                "progress": exp.progress if exp else 0,
                "total": exp.total_samples if exp else 0,
            })
        return jsonify({"error": "Cannot add rating"}), 400

    @app.route("/api/experiments/<experiment_id>/skip", methods=["POST"])
    def skip_entity(experiment_id: str):
        store = _get_store(app)
        data = request.json or {}
        rating = Rating(
            entity_id=data.get("entity_id", ""),
            source=data.get("source", ""),
            score=None,
            response_time_ms=None,
            skipped=True,
        )
        if store.add_rating(experiment_id, rating):
            return jsonify({"ok": True})
        return jsonify({"error": "Cannot skip"}), 400

    @app.route("/api/experiments/<experiment_id>/end", methods=["POST"])
    def end_experiment(experiment_id: str):
        store = _get_store(app)
        if store.finish_experiment(experiment_id):
            return jsonify({"ok": True})
        return jsonify({"error": "Experiment not found"}), 404

    @app.route("/api/experiments/<experiment_id>/results")
    def get_results(experiment_id: str):
        store = _get_store(app)
        exp = store.get(experiment_id)
        if not exp:
            return jsonify({"error": "Experiment not found"}), 404
        return jsonify(_compute_results(exp))

    # --- Entity Generation ---

    @app.route("/api/experiments/<experiment_id>/next")
    def next_entity(experiment_id: str):
        """Generate the next entity for an experiment.

        Picks a source (model or real) based on the experiment config,
        generates the entity, and returns it as SSE stream or JSON.

        For now, returns JSON directly. SSE streaming with blinding
        will be added in Phase 4.
        """
        store = _get_store(app)
        generator = app.config.get("generator")
        exp = store.get(experiment_id)

        if not exp:
            return jsonify({"error": "Experiment not found"}), 404
        if not exp.started:
            return jsonify({"error": "Experiment not started"}), 400
        if exp.finished:
            return jsonify({"error": "Experiment already finished"}), 400
        if not generator:
            return jsonify({"error": "No generator available"}), 500

        # Pick next source: round-robin across splits
        splits = []
        for m in exp.models:
            splits.append(("model", m.algorithm, m.timestamp))
        if exp.include_real:
            splits.append(("real", None, None))

        if not splits:
            return jsonify({"error": "No splits configured"}), 400

        # Round-robin based on progress
        split_idx = exp.progress % len(splits)
        # Shuffle within each full round
        if exp.progress % len(splits) == 0:
            random.shuffle(splits)
        source_type, alg, version = splits[split_idx]

        entity_id = str(uuid.uuid4())
        try:
            if source_type == "real":
                entity = generator.generate_entity_from_real(exp.view)
                source = "real"
            else:
                entity = generator.generate_entity_from_model(
                    exp.view, alg, version
                )
                source = f"{alg}_{version or 'latest'}"
        except Exception:
            logger.exception("Entity generation failed")
            return jsonify({"error": "Entity generation failed"}), 500

        entity_json = json.dumps(entity, indent=2)

        return jsonify({
            "entity_id": entity_id,
            "source": source,
            "entity": entity,
            "entity_json": entity_json,
        })

    # TODO: Phase 4 - SSE streaming version of next_entity
    # TODO: Phase 4 - Profiling endpoint


def _experiment_summary(exp: Experiment) -> dict:
    return {
        "id": exp.id,
        "name": exp.name,
        "view": exp.view,
        "num_models": len(exp.models),
        "include_real": exp.include_real,
        "blind": exp.blind,
        "samples_per_split": exp.samples_per_split,
        "total_samples": exp.total_samples,
        "progress": exp.progress,
        "finished": exp.finished,
        "started": exp.started,
        "created_at": exp.created_at,
    }


def _experiment_detail(exp: Experiment) -> dict:
    from dataclasses import asdict

    return {
        **_experiment_summary(exp),
        "models": [asdict(m) for m in exp.models],
        "timing_params": asdict(exp.timing_params) if exp.timing_params else None,
        "tutorial": exp.tutorial,
        "ratings": [asdict(r) for r in exp.ratings],
    }


def _compute_results(exp: Experiment) -> dict:
    """Compute aggregated results for an experiment."""
    from collections import defaultdict

    by_source: dict[str, list[int]] = defaultdict(list)
    for r in exp.ratings:
        if not r.skipped and r.score is not None:
            by_source[r.source].append(r.score)

    results = {}
    for source, scores in by_source.items():
        dist = {i: scores.count(i) for i in range(1, 6)}
        results[source] = {
            "count": len(scores),
            "mean": sum(scores) / len(scores) if scores else 0,
            "distribution": dist,
        }

    return {
        "experiment_id": exp.id,
        "name": exp.name,
        "view": exp.view,
        "total_rated": sum(1 for r in exp.ratings if not r.skipped),
        "total_skipped": sum(1 for r in exp.ratings if r.skipped),
        "by_source": results,
    }
