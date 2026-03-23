import { useCallback, useEffect, useRef, useState } from "react";
import {
  rateEntity,
  skipEntity,
  endExperiment,
} from "../api";
import type { ExperimentDetail, SSEEvent } from "../api";

interface Props {
  experiment: ExperimentDetail;
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

export default function EvaluationPage({ experiment, onFinished }: Props) {
  const [prettyJson, setPrettyJson] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [progress, setProgress] = useState(experiment.progress);
  const [entityId, setEntityId] = useState("");
  const [source, setSource] = useState("");
  const [streamProgress, setStreamProgress] = useState(0);
  const streamCompleteTime = useRef(0);
  const eventSourceRef = useRef<EventSource | null>(null);

  const total = experiment.total_samples;

  const startStreaming = useCallback(() => {
    setPrettyJson("");
    setStreaming(true);
    setStreamProgress(0);
    setEntityId("");
    setSource("");

    const es = new EventSource(
      `/api/experiments/${experiment.id}/next`
    );
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      const data: SSEEvent = JSON.parse(event.data);

      switch (data.type) {
        case "start":
          setEntityId(data.entity_id);
          break;
        case "token":
          if (data.pretty) {
            setPrettyJson(data.pretty);
          }
          setStreamProgress(data.progress);
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
  }, [experiment.id]);

  // Start streaming first entity on mount
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
      entityId,
      source,
      score,
      responseTimeMs
    );
    setProgress((p) => p + 1);

    if (progress + 1 >= total) {
      await endExperiment(experiment.id);
      onFinished();
    } else {
      startStreaming();
    }
  };

  const handleSkip = async () => {
    await skipEntity(experiment.id, entityId, source);
    setProgress((p) => p + 1);

    if (progress + 1 >= total) {
      await endExperiment(experiment.id);
      onFinished();
    } else {
      startStreaming();
    }
  };

  const handleEnd = async () => {
    eventSourceRef.current?.close();
    await endExperiment(experiment.id);
    onFinished();
  };

  return (
    <div className="page eval-page">
      <header className="header">
        <div className="header-left">
          <img src="/logo.svg" alt="Pasteur" className="header-logo" />
          <h1>LITMUS</h1>
          <span className="exp-name">{experiment.name}</span>
        </div>
        <div className="header-center">
          <span className="progress-text">
            {progress + 1} / {total}
          </span>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${(progress / total) * 100}%` }}
            />
          </div>
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

      {/* Entity display */}
      <div className="entity-container">
        {streaming && (
          <div className="stream-indicator">
            <div
              className="stream-bar"
              style={{ width: `${streamProgress * 100}%` }}
            />
          </div>
        )}
        <pre className="entity-json">{prettyJson || "Loading..."}</pre>
      </div>

      {/* Likert rating bar */}
      <div className="rating-bar">
        {LIKERT_LABELS.map((label, i) => {
          const score = i + 1;
          return (
            <button
              key={score}
              className="rating-btn"
              style={{
                backgroundColor: streaming
                  ? "#ccc"
                  : LIKERT_COLORS[i],
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
