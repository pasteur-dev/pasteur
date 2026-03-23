import { useEffect, useState } from "react";
import {
  createRun,
  deleteRun,
  fetchExperiment,
  resumeRun,
} from "../api";
import type { ExperimentDetail, RunSummary } from "../api";
import ModelLabel from "../components/ModelLabel";
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
  const [runName, setRunName] = useState(generateRunName);
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

  const handleResumeRun = async (run: RunSummary) => {
    if (run.finished) {
      await resumeRun(exp.id, run.id);
    }
    const updated = await fetchExperiment(exp.id);
    onResumeRun(updated, run);
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
                <ModelLabel key={i} model={m} showTimestamp />
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
          {exp.runs && exp.runs.length > 0 && (
            <section className="card">
              <h2>Runs ({exp.runs.length})</h2>
              <div className="experiment-list">
                {exp.runs.map((run) => (
                  <div
                    key={run.id}
                    className="experiment-item"
                    onClick={() => {
                      const canResume = !run.finished || run.progress < run.total_samples;
                      if (canResume) handleResumeRun(run);
                    }}
                    style={{
                      cursor: (!run.finished || run.progress < run.total_samples) ? "pointer" : "default",
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
                      {run.started && (!run.finished || run.progress < run.total_samples) && (
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

const ADJECTIVES = [
  "swift", "keen", "bold", "calm", "fair", "wise", "warm", "cool",
  "bright", "sharp", "clear", "quick", "steady", "gentle", "vivid",
];
const NOUNS = [
  "falcon", "cedar", "prism", "atlas", "helix", "quartz", "lotus",
  "cipher", "drift", "nexus", "pulse", "spark", "orbit", "slate",
];

function generateRunName(): string {
  const adj = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
  const noun = NOUNS[Math.floor(Math.random() * NOUNS.length)];
  const num = Math.floor(Math.random() * 100);
  return `${adj}-${noun}-${num}`;
}
