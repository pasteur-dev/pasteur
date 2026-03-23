import { useCallback, useEffect, useRef, useState } from "react";
import { rateEntity, skipEntity, endRun } from "../api";
import type { ExperimentDetail } from "../api";
import EntityCard from "../components/EntityCard";

interface Props {
  experiment: ExperimentDetail;
  runId: string;
  onFinished: () => void;
}

const LIKERT_LABELS = [
  "Clearly artificial",
  "Likely artificial",
  "Uncertain",
  "Likely real",
  "Indistinguishable",
];

const LIKERT_COLORS = [
  "#ff2244",
  "#ff6633",
  "#cc9900",
  "#44bb55",
  "#00dd77",
];

export default function EvaluationPage({
  experiment,
  runId,
  onFinished,
}: Props) {
  const [entity, setEntity] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const currentRun = experiment.runs?.find((r) => r.id === runId);
  const [progress, setProgress] = useState(currentRun?.progress ?? 0);
  const [entityId, setEntityId] = useState("");
  const [source, setSource] = useState("");
  const loadCompleteTime = useRef(0);

  const total = experiment.total_samples;

  const fetchNext = useCallback(async () => {
    setEntity(null);
    setLoading(true);
    setEntityId("");
    setSource("");

    try {
      const res = await fetch(
        `/api/experiments/${experiment.id}/runs/${runId}/next`
      );
      const data = await res.json();
      if (data.error) {
        console.error("Entity generation error:", data.error);
        return;
      }
      setEntityId(data.entity_id);
      setSource(data.source);
      setEntity(data.entity);
    } catch (err) {
      console.error("Failed to fetch entity:", err);
    } finally {
      setLoading(false);
      loadCompleteTime.current = performance.now();
    }
  }, [experiment.id, runId]);

  useEffect(() => {
    fetchNext();
  }, [fetchNext]);

  const handleRate = async (score: number) => {
    const responseTimeMs = Math.round(
      performance.now() - loadCompleteTime.current
    );
    await rateEntity(
      experiment.id,
      runId,
      entityId,
      source,
      score,
      responseTimeMs
    );
    const newProgress = progress + 1;
    setProgress(newProgress);

    if (newProgress >= total) {
      await endRun(experiment.id, runId);
      onFinished();
    } else {
      fetchNext();
    }
  };

  const handleSkip = async () => {
    await skipEntity(experiment.id, runId, entityId, source);
    const newProgress = progress + 1;
    setProgress(newProgress);

    if (newProgress >= total) {
      await endRun(experiment.id, runId);
      onFinished();
    } else {
      fetchNext();
    }
  };

  const handleEnd = async () => {
    await endRun(experiment.id, runId);
    onFinished();
  };

  return (
    <div className="page eval-page">
      <header className="header">
        <div className="header-left">
          <a href="#/" className="header-brand">
            <img src="/logo.svg" alt="Pasteur" className="header-logo" />
            <h1>LITMUS</h1>
          </a>
          <span className="exp-name">{experiment.name}</span>
        </div>
        <div className="header-center">
          <span className="progress-text">
            {progress + 1} / {total}
          </span>
        </div>
        <div className="header-right">
          <button
            className="btn btn-small"
            onClick={handleSkip}
            disabled={loading}
          >
            Skip
          </button>
          <button className="btn btn-small btn-danger" onClick={handleEnd}>
            End
          </button>
        </div>
      </header>

      <div className="entity-container">
        {entity ? (
          <EntityCard data={entity} streaming={loading} />
        ) : (
          <div className="entity-loading">Generating entity...</div>
        )}
      </div>

      <div className="rating-bar">
        {LIKERT_LABELS.map((label, i) => {
          const score = i + 1;
          return (
            <button
              key={score}
              className="rating-btn"
              style={{
                backgroundColor: loading ? "#1a1a2e" : LIKERT_COLORS[i],
              }}
              disabled={loading}
              onClick={() => handleRate(score)}
              title={label}
            >
              <span className="rating-score">{score}</span>
              <span className="rating-label">{label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
