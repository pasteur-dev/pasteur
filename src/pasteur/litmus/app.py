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
        return jsonify(_run_detail(run, store.get_experiment(eid))), 201

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
        exp = store.get_experiment(eid)
        if not exp:
            return jsonify({"error": "Experiment not found"}), 404
        return jsonify(_compute_results(exp))

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
        if source_type == "real":
            entity = generator.generate_entity_from_real(exp.view)
            source = "real"
        else:
            entity = generator.generate_entity_from_model(
                exp.view, alg, version
            )
            source = f"{alg}_{version or 'latest'}"
        return entity, source

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
            entity, source = _generate_entity(
                generator, exp, source_type, alg, version
            )
        except Exception:
            logger.exception("Entity generation failed")
            return jsonify({"error": "Entity generation failed"}), 500

        return jsonify({
            "entity_id": entity_id,
            "source": source,
            "entity": entity,
        })


# --- Serialization helpers ---

def _exp_summary(exp: Experiment) -> dict:
    return {
        "id": exp.id,
        "name": exp.name,
        "view": exp.view,
        "num_models": len(exp.models),
        "include_real": exp.include_real,
        "blind": exp.blind,
        "samples_per_split": exp.samples_per_split,
        "total_samples": exp.total_samples,
        "num_runs": len(exp.runs),
        "created_at": exp.created_at,
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


def _compute_results(exp: Experiment) -> dict:
    from collections import defaultdict

    by_source: dict[str, list[int]] = defaultdict(list)
    for run in exp.runs:
        if run.tutorial:
            continue
        for r in run.ratings:
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
        "by_source": results,
    }
