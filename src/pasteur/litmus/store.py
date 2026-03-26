"""JSON-backed persistence for LITMUS experiments and runs."""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from threading import Event, Lock

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
    schedule: list[list] = field(default_factory=list)
    """Pre-determined sample order: list of [source_key, entity_index].
    Generated once at experiment creation, shared across all runs."""

    tutorial_schedule: list[list] = field(default_factory=list)
    """Tutorial schedule: 2 per split, ordered real first then synth (unshuffled)."""

    source_seeds: dict[str, int] = field(default_factory=dict)
    """Per-source RNG seeds for deterministic pool shuffling."""

    tutorial_seeds: dict[str, int] = field(default_factory=dict)
    """Per-source RNG seeds for tutorial pool shuffling."""

    TUTORIAL_PER_SPLIT = 2

    @property
    def num_splits(self) -> int:
        return len(self.models) + (1 if self.include_real else 0)

    @property
    def total_samples(self) -> int:
        return self.num_splits * self.samples_per_split

    @property
    def tutorial_total(self) -> int:
        return self.num_splits * self.TUTORIAL_PER_SPLIT

    def generate_schedule(self):
        """Generate a deterministic schedule seeded by experiment ID.

        Creates a shuffled sequence of (source_key, entity_index) pairs.
        Each source gets exactly samples_per_split entries.
        Entity indices are stable — they index into the entity pool for
        that source.
        """
        import hashlib
        import random as _random

        seed = int(hashlib.sha256(self.id.encode()).hexdigest(), 16) % (2**32)
        rng = _random.Random(seed)

        # Build source keys
        sources = []
        for m in self.models:
            sources.append(f"{m.algorithm}_{m.timestamp or 'latest'}")
        if self.include_real:
            sources.append("real")

        # For each source, generate a per-source seed. At entity retrieval
        # time the pool is shuffled with this seed to pick unique entities.
        source_seeds = {}
        for source in sources:
            source_seeds[source] = rng.randint(0, 2**32 - 1)

        # Build schedule entries: [source_key, nth_unique] where nth_unique
        # is the 0-based index into the deduplicated sequence for that source.
        counters: dict[str, int] = {s: 0 for s in sources}
        schedule = []
        for source in sources:
            for _ in range(self.samples_per_split):
                schedule.append([source, counters[source]])
                counters[source] += 1

        # Shuffle the full schedule deterministically
        rng.shuffle(schedule)
        self.schedule = schedule
        self.source_seeds = source_seeds

        # Tutorial schedule: 2 per split, real first then synth, not shuffled
        tutorial_rng = _random.Random(seed + 1)
        tutorial_seeds = {}
        tutorial = []
        # Real first
        if self.include_real:
            tutorial_seeds["real"] = tutorial_rng.randint(0, 2**32 - 1)
            for i in range(self.TUTORIAL_PER_SPLIT):
                tutorial.append(["real", i])
        # Then each synth model
        for m in self.models:
            key = f"{m.algorithm}_{m.timestamp or 'latest'}"
            tutorial_seeds[key] = tutorial_rng.randint(0, 2**32 - 1)
            for i in range(self.TUTORIAL_PER_SPLIT):
                tutorial.append([key, i])
        self.tutorial_schedule = tutorial
        self.tutorial_seeds = tutorial_seeds


def _experiment_to_dict(exp: Experiment) -> dict:
    return asdict(exp)


def _experiment_from_dict(d: dict) -> Experiment:
    models = [ModelRef(**m) for m in d.pop("models", [])]
    raw_runs = d.pop("runs", [])
    runs = []
    for r in raw_runs:
        ratings = [Rating(**rt) for rt in r.pop("ratings", [])]
        runs.append(Run(ratings=ratings, **r))
    d.setdefault("blind", True)
    d.setdefault("schedule", [])
    # Filter to known fields to ignore stale/removed keys
    known = {f.name for f in fields(Experiment)} - {"models", "runs"}
    d = {k: v for k, v in d.items() if k in known}
    exp = Experiment(
        models=models,
        runs=runs,
        **d,
    )
    # Auto-generate schedules for legacy experiments
    if not exp.schedule or not exp.tutorial_schedule or not exp.source_seeds:
        exp.generate_schedule()
        exp._needs_save = True
    return exp


class ExperimentStore:
    """Manages experiment persistence as JSON files in a directory."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir) / "litmus" / "experiments"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: dict[str, Experiment] = {}
        self._lock = Lock()
        self._version = 0
        self._version_event = Event()
        self._load_all()

    @property
    def version(self) -> int:
        return self._version

    def _bump_version(self):
        """Increment version and wake any long-polling waiters."""
        self._version += 1
        self._version_event.set()
        self._version_event = Event()

    def wait_for_change(self, known_version: int, timeout: float = 10.0) -> int:
        """Block until version changes or timeout. Returns current version."""
        if self._version != known_version:
            return self._version
        self._version_event.wait(timeout=timeout)
        return self._version

    def _path(self, experiment_id: str) -> Path:
        return self.data_dir / f"{experiment_id}.json"

    def _load_all(self):
        for f in self.data_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                exp = _experiment_from_dict(data)
                self._experiments[exp.id] = exp
                if getattr(exp, "_needs_save", False):
                    self._save(exp)
                    del exp._needs_save
                logger.info(f"Loaded experiment: {exp.id} ({exp.name or 'unnamed'})")
            except Exception:
                logger.exception(f"Failed to load experiment from {f}")

    def _save(self, exp: Experiment):
        with open(self._path(exp.id), "w") as f:
            json.dump(_experiment_to_dict(exp), f, indent=2)
        self._bump_version()

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
            exp.generate_schedule()
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
            self._bump_version()
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
                    sched = exp.tutorial_schedule if run.tutorial else exp.schedule
                    if run.progress >= len(sched):
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

