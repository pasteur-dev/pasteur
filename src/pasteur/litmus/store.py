"""JSON-backed persistence for LITMUS experiments and ratings."""

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
class TimingParams:
    t_mare: float = 0.0
    tps_0: float = 0.0
    gamma: float = 0.0


@dataclass
class Experiment:
    id: str
    view: str
    models: list[ModelRef]
    include_real: bool
    blind: bool
    samples_per_split: int
    name: str = ""
    timing_params: TimingParams | None = None
    tutorial: bool = False
    created_at: str = ""
    finished: bool = False
    started: bool = False
    ratings: list[Rating] = field(default_factory=list)

    @property
    def num_splits(self) -> int:
        return len(self.models) + (1 if self.include_real else 0)

    @property
    def total_samples(self) -> int:
        return self.num_splits * self.samples_per_split

    @property
    def progress(self) -> int:
        return len(self.ratings)


def _experiment_to_dict(exp: Experiment) -> dict:
    d = asdict(exp)
    return d


def _experiment_from_dict(d: dict) -> Experiment:
    models = [ModelRef(**m) for m in d.pop("models", [])]
    ratings = [Rating(**r) for r in d.pop("ratings", [])]
    tp = d.pop("timing_params", None)
    timing_params = TimingParams(**tp) if tp else None
    return Experiment(
        models=models,
        ratings=ratings,
        timing_params=timing_params,
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

    def create(
        self,
        view: str,
        models: list[ModelRef],
        include_real: bool,
        blind: bool,
        samples_per_split: int,
    ) -> Experiment:
        with self._lock:
            exp = Experiment(
                id=str(uuid.uuid4()),
                view=view,
                models=models,
                include_real=include_real,
                blind=blind,
                samples_per_split=samples_per_split,
                created_at=datetime.now().isoformat(),
            )
            self._experiments[exp.id] = exp
            self._save(exp)
            logger.info(f"Created experiment {exp.id}")
            return exp

    def get(self, experiment_id: str) -> Experiment | None:
        return self._experiments.get(experiment_id)

    def list_all(self) -> list[Experiment]:
        return sorted(
            self._experiments.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )

    def update(self, exp: Experiment):
        with self._lock:
            self._experiments[exp.id] = exp
            self._save(exp)

    def delete(self, experiment_id: str) -> bool:
        with self._lock:
            if experiment_id not in self._experiments:
                return False
            del self._experiments[experiment_id]
            path = self._path(experiment_id)
            if path.exists():
                os.remove(path)
            logger.info(f"Deleted experiment {experiment_id}")
            return True

    def add_rating(self, experiment_id: str, rating: Rating) -> bool:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp or exp.finished:
                return False
            exp.ratings.append(rating)
            self._save(exp)
            return True

    def start_experiment(
        self, experiment_id: str, name: str, tutorial: bool
    ) -> bool:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            exp.name = name
            exp.tutorial = tutorial
            exp.started = True
            self._save(exp)
            return True

    def finish_experiment(self, experiment_id: str) -> bool:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            exp.finished = True
            self._save(exp)
            return True

    def set_timing_params(
        self, experiment_id: str, params: TimingParams
    ) -> bool:
        with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            exp.timing_params = params
            self._save(exp)
            return True
