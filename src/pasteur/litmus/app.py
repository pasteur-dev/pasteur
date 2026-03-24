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
    if total == 0:
        return {
            "pretty_name": pretty_name,
            "count": 0,
            "mean": 0,
            "std": 0,
            "median": 0,
            "distribution": dist,
        }
    # Expand distribution to list of scores
    scores = []
    for k, v in dist.items():
        scores.extend([k] * v)
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance**0.5
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    median = (
        sorted_scores[n // 2]
        if n % 2 == 1
        else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    )
    return {
        "pretty_name": pretty_name,
        "count": total,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "median": median,
        "distribution": dist,
    }


def _compute_inter_rater(exp, source_order, pretty_names) -> dict | None:
    """Compute Krippendorff's alpha for inter-rater agreement.

    Uses ordinal metric. Requires >=2 non-tutorial runs.
    Returns per-source and overall alpha, or None if insufficient data.
    """
    from collections import defaultdict

    non_tutorial_runs = [r for r in exp.runs if not r.tutorial and r.ratings]
    if len(non_tutorial_runs) < 2:
        return None

    # Build reliability matrix: rows=raters, cols=items
    # Items are identified by their index within each run
    # For per-source alpha, group by source
    per_source = {}

    for source in source_order:
        # For each run, collect ratings for this source in order
        source_ratings: list[list[int]] = []
        for run in non_tutorial_runs:
            run_source_scores = [
                r.score
                for r in run.ratings
                if r.source == source and not r.skipped and r.score is not None
            ]
            source_ratings.append(run_source_scores)

        # Need same number of items per rater for simple alpha
        min_items = min(len(sr) for sr in source_ratings) if source_ratings else 0
        if min_items < 2:
            continue

        # Truncate to min_items
        truncated = [sr[:min_items] for sr in source_ratings]
        alpha = _krippendorff_alpha_ordinal(truncated)
        if alpha is not None:
            per_source[source] = {
                "pretty_name": pretty_names.get(source, source),
                "alpha": round(alpha, 3),
                "n_raters": len(non_tutorial_runs),
                "n_items": min_items,
            }

    # Overall alpha across all sources
    all_truncated = []
    for run in non_tutorial_runs:
        run_scores = [
            r.score
            for r in run.ratings
            if not r.skipped and r.score is not None
        ]
        all_truncated.append(run_scores)

    min_all = min(len(rs) for rs in all_truncated) if all_truncated else 0
    overall_alpha = None
    if min_all >= 2:
        truncated_all = [rs[:min_all] for rs in all_truncated]
        overall_alpha = _krippendorff_alpha_ordinal(truncated_all)

    return {
        "overall": round(overall_alpha, 3) if overall_alpha is not None else None,
        "n_raters": len(non_tutorial_runs),
        "per_source": list(per_source.values()),
    }


def _krippendorff_alpha_ordinal(data: list[list[int]]) -> float | None:
    """Compute Krippendorff's alpha with ordinal metric.

    data: list of rater scores, each a list of scores for items.
    All sublists must be the same length.
    """
    n_raters = len(data)
    n_items = len(data[0]) if data else 0
    if n_raters < 2 or n_items < 2:
        return None

    # Collect all values
    all_values = sorted(set(v for row in data for v in row))
    if len(all_values) < 2:
        return 1.0  # Perfect agreement if only one value

    # Ordinal distance: squared cumulative-proportion distance
    val_to_idx = {v: i for i, v in enumerate(all_values)}
    n_vals = len(all_values)

    # Frequency of each value
    freq = [0] * n_vals
    for row in data:
        for v in row:
            freq[val_to_idx[v]] += 1
    total_n = sum(freq)

    # Ordinal metric: d(c,k) = sum_{g=c}^{k} (f_g) - (f_c + f_k)/2
    def ordinal_dist_sq(c_idx, k_idx):
        if c_idx == k_idx:
            return 0.0
        lo, hi = min(c_idx, k_idx), max(c_idx, k_idx)
        s = sum(freq[g] for g in range(lo, hi + 1)) - (freq[lo] + freq[hi]) / 2
        return s * s

    # Observed disagreement
    Do = 0.0
    for item in range(n_items):
        item_vals = [data[r][item] for r in range(n_raters)]
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                Do += ordinal_dist_sq(
                    val_to_idx[item_vals[i]], val_to_idx[item_vals[j]]
                )
    n_pairs_per_item = n_raters * (n_raters - 1) / 2
    Do /= n_items * n_pairs_per_item

    # Expected disagreement
    De = 0.0
    for c in range(n_vals):
        for k in range(c + 1, n_vals):
            De += freq[c] * freq[k] * ordinal_dist_sq(c, k)
    De *= 2.0 / (total_n * (total_n - 1))

    if De == 0:
        return 1.0
    return 1.0 - Do / De


def _compute_human_llm_comparison(
    human_results, llm_results, source_order, pretty_names
) -> dict | None:
    """Compute per-source mean difference and rank comparison between human and LLM."""
    if not human_results or not llm_results:
        return None

    human_means = {r["source"]: r["mean"] for r in human_results}
    llm_means = {r["source"]: r["mean"] for r in llm_results}

    common = [s for s in source_order if s in human_means and s in llm_means]
    if len(common) < 2:
        return None

    # Per-source mean difference (human - LLM)
    per_source = []
    for s in common:
        diff = human_means[s] - llm_means[s]
        per_source.append({
            "source": s,
            "pretty_name": pretty_names.get(s, s),
            "human_mean": round(human_means[s], 2),
            "llm_mean": round(llm_means[s], 2),
            "diff": round(diff, 2),
        })

    # Rank comparison
    human_ranked = sorted(common, key=lambda s: human_means[s], reverse=True)
    llm_ranked = sorted(common, key=lambda s: llm_means[s], reverse=True)

    human_ranking = [pretty_names.get(s, s) for s in human_ranked]
    llm_ranking = [pretty_names.get(s, s) for s in llm_ranked]

    # Check if rankings match
    rank_match = human_ranked == llm_ranked

    return {
        "per_source": per_source,
        "human_ranking": human_ranking,
        "llm_ranking": llm_ranking,
        "rank_match": rank_match,
        "n_sources": len(common),
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

    # Response times per source (ms -> seconds)
    response_times: dict[str, list[float]] = defaultdict(list)
    for run in exp.runs:
        if run.tutorial:
            continue
        for r in run.ratings:
            if not r.skipped and r.response_time_ms is not None:
                response_times[r.source].append(r.response_time_ms / 1000.0)

    response_time_stats = []
    for source in source_order:
        times = response_times.get(source, [])
        if not times:
            continue
        response_time_stats.append({
            "source": source,
            "pretty_name": pretty_names.get(source, source),
            "mean": round(sum(times) / len(times), 2),
            "count": len(times),
            "times": [round(t, 2) for t in times],
        })

    # Inter-rater agreement (Krippendorff's alpha, ordinal)
    inter_rater = _compute_inter_rater(exp, source_order, pretty_names)

    # Human vs LLM comparison (per-source mean diff + rank)
    human_llm = _compute_human_llm_comparison(
        human_results, llm_results, source_order, pretty_names
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
        "response_times": response_time_stats,
        "inter_rater": inter_rater,
        "human_llm_comparison": human_llm,
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

    # Response times per source
    response_times: dict[str, list[float]] = defaultdict(list)
    for r in run.ratings:
        if not r.skipped and r.response_time_ms is not None:
            response_times[r.source].append(r.response_time_ms / 1000.0)

    response_time_stats = []
    for source in source_order:
        times = response_times.get(source, [])
        if not times:
            continue
        response_time_stats.append({
            "source": source,
            "pretty_name": pretty_names.get(source, source),
            "mean": round(sum(times) / len(times), 2),
            "count": len(times),
            "times": [round(t, 2) for t in times],
        })

    return {
        "run_id": run.id,
        "name": run.name,
        "total_rated": total_rated,
        "total_skipped": total_skipped,
        "by_source": results,
        "response_times": response_time_stats,
    }
