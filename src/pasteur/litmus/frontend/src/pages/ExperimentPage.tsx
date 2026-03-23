import { useEffect, useState } from "react";
import {
  createRun,
  deleteRun,
  fetchExperiment,
} from "../api";
import type { ExperimentDetail, RunSummary } from "../api";
import ResultsPanel from "./ResultsPanel";

interface Props {
  experiment: ExperimentDetail;
  onStartRun: (exp: ExperimentDetail, run: RunSummary) => void;
  onResumeRun: (exp: ExperimentDetail, run: RunSummary) => void;
  onBack: () => void;
}

export default function ExperimentPage({
  experiment: initialExp,
  onStartRun,
  onResumeRun,
  onBack,
}: Props) {
  const [exp, setExp] = useState(initialExp);
  const [runName, setRunName] = useState("");
  const [tutorial, setTutorial] = useState(false);

  // Refresh experiment data
  const refresh = async () => {
    const updated = await fetchExperiment(exp.id);
    setExp(updated);
  };

  useEffect(() => {
    refresh();
  }, [exp.id]);

  const handleNewRun = async () => {
    const run = await createRun(exp.id, runName, tutorial);
    const updated = await fetchExperiment(exp.id);
    onStartRun(updated, run);
  };

  const handleDeleteRun = async (runId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await deleteRun(exp.id, runId);
    refresh();
  };

  const handleResumeRun = (run: RunSummary) => {
    onResumeRun(exp, run);
  };

  return (
    <div className="page experiment-page">
      <header className="header">
        <a href="/" className="header-brand">
          <img src="/logo.svg" alt="Pasteur" className="header-logo" />
          <h1>LITMUS</h1>
        </a>
        <button className="btn btn-small" onClick={onBack}>
          Back
        </button>
      </header>

      <div className="setup-layout">
        {/* Left: experiment info + new run */}
        <div className="setup-left">
          <section className="card">
            <h2>{exp.name || "Unnamed Experiment"}</h2>
            <div className="exp-summary">
              <strong>{exp.view}</strong> &middot;{" "}
              {exp.num_models} model{exp.num_models !== 1 ? "s" : ""}
              {exp.include_real ? " + real data" : ""} &middot;{" "}
              {exp.total_samples} samples per run
              {exp.blind ? " (blinded)" : ""}
            </div>

            <div className="exp-models">
              {exp.models.map((m, i) => (
                <span key={i} className="model-tag">
                  {m.algorithm}
                  <span className="model-tag-version">
                    {formatTimestamp(m.timestamp)}
                  </span>
                </span>
              ))}
              {exp.include_real && (
                <span className="model-tag model-tag-real">real</span>
              )}
            </div>
          </section>

          <section className="card">
            <h2>New Run</h2>
            <label>
              Participant / session name
              <input
                type="text"
                value={runName}
                onChange={(e) => setRunName(e.target.value)}
                placeholder="Reviewing Medical Patients (John Doe)"
              />
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={tutorial}
                onChange={(e) => setTutorial(e.target.checked)}
              />
              Tutorial first (2 samples per model, unblinded, results
              discarded)
            </label>

            <button className="btn btn-primary" onClick={handleNewRun}>
              Start Run
            </button>
          </section>

          {/* Run history */}
          {exp.runs.length > 0 && (
            <section className="card">
              <h2>Runs ({exp.runs.length})</h2>
              <div className="experiment-list">
                {exp.runs.map((run) => (
                  <div
                    key={run.id}
                    className="experiment-item"
                    onClick={() =>
                      !run.finished && handleResumeRun(run)
                    }
                    style={{
                      cursor: run.finished ? "default" : "pointer",
                    }}
                  >
                    <div className="exp-info">
                      <strong>{run.name || "Unnamed run"}</strong>
                      <span>
                        {run.progress}/{run.total_samples}
                        {run.finished
                          ? " (done)"
                          : run.started
                            ? " (in progress)"
                            : ""}
                        {run.tutorial ? " [tutorial]" : ""}
                      </span>
                    </div>
                    <div className="exp-actions">
                      {run.started && !run.finished && (
                        <button
                          className="btn btn-small btn-primary"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleResumeRun(run);
                          }}
                        >
                          Resume
                        </button>
                      )}
                      <button
                        className="btn btn-small btn-danger"
                        onClick={(e) => handleDeleteRun(run.id, e)}
                        title="Delete run"
                      >
                        &#128465;
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>

        {/* Right: results */}
        <div className="setup-right">
          <ResultsPanel experimentId={exp.id} />
        </div>
      </div>
    </div>
  );
}

function formatTimestamp(ts: string): string {
  return ts
    .replace(/T/, " ")
    .replace(/\.\d+Z$/, "")
    .replace(/\./g, ":");
}
