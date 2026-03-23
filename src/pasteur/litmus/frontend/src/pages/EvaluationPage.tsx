import { useCallback, useEffect, useRef, useState } from "react";
import { rateEntity, skipEntity, endRun } from "../api";
import type { ExperimentDetail, SSEEvent } from "../api";
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
  const [streaming, setStreaming] = useState(false);
  const currentRun = experiment.runs?.find((r) => r.id === runId);
  const [progress, setProgress] = useState(currentRun?.progress ?? 0);
  const [entityId, setEntityId] = useState("");
  const [source, setSource] = useState("");
  const streamCompleteTime = useRef(0);
  const eventSourceRef = useRef<EventSource | null>(null);
  const accumulatedRef = useRef("");

  const total = experiment.total_samples;

  const startStreaming = useCallback(() => {
    setEntity(null);
    setStreaming(true);
    setEntityId("");
    setSource("");
    accumulatedRef.current = "";

    const es = new EventSource(
      `/api/experiments/${experiment.id}/runs/${runId}/next`
    );
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const data: SSEEvent = JSON.parse(event.data);

      switch (data.type) {
        case "start":
          setEntityId(data.entity_id);
          break;
        case "token":
          // Try to parse the accumulated JSON for progressive rendering
          if (data.pretty) {
            try {
              setEntity(JSON.parse(data.pretty));
            } catch {
              // partial parse failed, keep previous state
            }
          }
          break;
        case "done":
          setSource(data.source);
          setStreaming(false);
          streamCompleteTime.current = performance.now();
          es.close();
          eventSourceRef.current = null;
          break;
      }
    };

    es.onerror = () => {
      setStreaming(false);
      es.close();
      eventSourceRef.current = null;
    };
  }, [experiment.id, runId]);

  useEffect(() => {
    startStreaming();
    return () => {
      eventSourceRef.current?.close();
    };
  }, [startStreaming]);

  const handleRate = async (score: number) => {
    const responseTimeMs = Math.round(
      performance.now() - streamCompleteTime.current
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
      startStreaming();
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
      startStreaming();
    }
  };

  const handleEnd = async () => {
    eventSourceRef.current?.close();
    await endRun(experiment.id, runId);
    onFinished();
  };

  return (
    <div className="page eval-page">
      <header className="header">
        <div className="header-left">
          <a href="/" className="header-brand">
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
            disabled={streaming}
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
          <EntityCard data={entity} streaming={streaming} />
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
                backgroundColor: streaming ? "#1a1a2e" : LIKERT_COLORS[i],
              }}
              disabled={streaming}
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
