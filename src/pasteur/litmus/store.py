"""JSON-backed persistence for LITMUS experiments and runs."""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class Rating:
    entity_id: str
    source: str  # e.g. "amalgam_e1", "mare_e10", "real"
    score: int | None  # 1-5 Likert, None if skipped
    response_time_ms: int | None
    skipped: bool = False


@dataclass
class ModelRef:
    algorithm: str
    timestamp: str
    overrides: dict = field(default_factory=dict)


@dataclass
class Run:
    """A single participant session within an experiment."""

    id: str
    name: str  # participant/session name
    tutorial: bool = False
    started: bool = False
    finished: bool = False
    created_at: str = ""
    ratings: list[Rating] = field(default_factory=list)

    @property
    def progress(self) -> int:
        return len(self.ratings)


@dataclass
class Experiment:
    """Top-level experiment configuration. Contains multiple runs."""

    id: str
    view: str
    models: list[ModelRef]
    include_real: bool
    blind: bool
    samples_per_split: int
    name: str = ""
    created_at: str = ""
    runs: list[Run] = field(default_factory=list)

    @property
    def num_splits(self) -> int:
        return len(self.models) + (1 if self.include_real else 0)

    @property
    def total_samples(self) -> int:
        return self.num_splits * self.samples_per_split


def _experiment_to_dict(exp: Experiment) -> dict:
    return asdict(exp)


def _experiment_from_dict(d: dict) -> Experiment:
    models = [ModelRef(**m) for m in d.pop("models", [])]
    raw_runs = d.pop("runs", [])
    runs = []
    for r in raw_runs:
        ratings = [Rating(**rt) for rt in r.pop("ratings", [])]
        runs.append(Run(ratings=ratings, **r))
    # Drop legacy fields
    d.pop("timing_params", None)
    d.pop("tutorial", None)
    d.pop("finished", None)
    d.setdefault("blind", True)
    return Experiment(
        models=models,
        runs=runs,
        **d,
    )


class ExperimentStore:
    """Manages experiment persistence as JSON files in a directory."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir) / "litmus" / "experiments"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: dict[str, Experiment] = {}
        self._lock = Lock()
        self._load_all()

    def _path(self, experiment_id: str) -> Path:
        return self.data_dir / f"{experiment_id}.json"

    def _load_all(self):
        for f in self.data_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                exp = _experiment_from_dict(data)
                self._experiments[exp.id] = exp
                logger.info(f"Loaded experiment: {exp.id} ({exp.name or 'unnamed'})")
            except Exception:
                logger.exception(f"Failed to load experiment from {f}")

    def _save(self, exp: Experiment):
        with open(self._path(exp.id), "w") as f:
            json.dump(_experiment_to_dict(exp), f, indent=2)

    # --- Experiment CRUD ---

    def create_experiment(
        self,
        name: str,
        view: str,
        models: list[ModelRef],
        include_real: bool,
        blind: bool,
        samples_per_split: int,
    ) -> Experiment:
        with self._lock:
            exp = Experiment(
                id=str(uuid.uuid4()),
                name=name,
                view=view,
                models=models,
                include_real=include_real,
                blind=blind,
                samples_per_split=samples_per_split,
                created_at=datetime.now().isoformat(),
            )
            self._experiments[exp.id] = exp
            self._save(exp)
            logger.info(f"Created experiment {exp.id}: {name}")
            return exp

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[Experiment]:
        return sorted(
            self._experiments.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )

    def update_experiment(self, exp: Experiment):
        with self._lock:
            self._experiments[exp.id] = exp
            self._save(exp)

    def delete_experiment(self, experiment_id: str) -> bool:
        with self._lock:
            if experiment_id not in self._experiments:
                return False
            del self._experiments[experiment_id]
            path = self._path(experiment_id)
            if path.exists():
                os.remove(path)
            logger.info(f"Deleted experiment {experiment_id}")
            return True

    # --- Run CRUD ---

    def create_run(
        self, experiment_id: str, name: str, tutorial: bool
    ) -> Run | None:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return None
            run = Run(
                id=str(uuid.uuid4()),
                name=name,
                tutorial=tutorial,
                started=True,
                created_at=datetime.now().isoformat(),
            )
            exp.runs.append(run)
            self._save(exp)
            logger.info(f"Created run {run.id} in experiment {experiment_id}")
            return run

    def get_run(self, experiment_id: str, run_id: str) -> tuple[Experiment, Run] | None:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None
        for run in exp.runs:
            if run.id == run_id:
                return exp, run
        return None

    def add_rating(self, experiment_id: str, run_id: str, rating: Rating) -> bool:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            for run in exp.runs:
                if run.id == run_id:
                    if run.finished:
                        return False
                    run.ratings.append(rating)
                    self._save(exp)
                    return True
            return False

    def undo_rating(self, experiment_id: str, run_id: str) -> Rating | None:
        """Remove and return the last rating from a run."""
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return None
            for run in exp.runs:
                if run.id == run_id:
                    if not run.ratings:
                        return None
                    rating = run.ratings.pop()
                    self._save(exp)
                    return rating
            return None

    def finish_run(self, experiment_id: str, run_id: str) -> bool:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            for run in exp.runs:
                if run.id == run_id:
                    run.finished = True
                    self._save(exp)
                    return True
            return False

    def resume_run(self, experiment_id: str, run_id: str) -> bool:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            for run in exp.runs:
                if run.id == run_id:
                    run.finished = False
                    self._save(exp)
                    return True
            return False

    def delete_run(self, experiment_id: str, run_id: str) -> bool:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            exp.runs = [r for r in exp.runs if r.id != run_id]
            self._save(exp)
            return True

