import { useEffect, useState } from "react";
import {
  createRun,
  deleteRun,
  fetchExperiment,
  fetchRunResults,
  resumeRun,
  updateExperiment,
} from "../api";
import type { ExperimentDetail, RunSummary, RunResults } from "../api";
import { ModelTable } from "../components/ModelLabel";
import NumberInput from "../components/NumberInput";
import RunResultsModal from "../components/RunResultsModal";
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
  const [runResults, setRunResults] = useState<RunResults | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [settingsSamples, setSettingsSamples] = useState(initialExp.samples_per_split);
  const [settingsBlind, setSettingsBlind] = useState(initialExp.blind);

  // Refresh experiment data
  const refresh = async () => {
    const updated = await fetchExperiment(exp.id);
    setExp(updated);
  };

  useEffect(() => {
    refresh();
  }, [exp.id]);

  const handleNewRun = async () => {
    const name = runName || generateRunName();
    const run = await createRun(exp.id, name, false);
    const updated = await fetchExperiment(exp.id);
    onStartRun(updated, run);
  };

  const handleDeleteRun = async (runId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await deleteRun(exp.id, runId);
    refresh();
  };

  const handleRunInfo = async (runId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const results = await fetchRunResults(exp.id, runId);
    setRunResults(results);
  };

  const handleResumeRun = async (run: RunSummary) => {
    if (run.finished) {
      await resumeRun(exp.id, run.id);
    }
    const updated = await fetchExperiment(exp.id);
    onResumeRun(updated, run);
  };

  const openSettings = () => {
    setSettingsSamples(exp.samples_per_split);
    setSettingsBlind(exp.blind);
    setShowSettings(true);
  };

  const saveSettings = async () => {
    const updated = await updateExperiment(exp.id, {
      samples_per_split: settingsSamples,
      blind: settingsBlind,
    });
    setExp(updated);
    setShowSettings(false);
  };

  return (
    <div className="page experiment-page">
      <header className="header">
        <a href="#/" className="header-brand">
          <img src="/logo.svg" alt="Pasteur" className="header-logo" />
          <h1>LITMUS</h1>
        </a>
        <div className="header-actions">
          <button
            className="btn btn-small btn-settings"
            onClick={openSettings}
            title="Experiment settings"
          >
            &#9881;
          </button>
          <button className="btn btn-small btn-undo" onClick={onBack}>
            Experiments
          </button>
        </div>
      </header>

      <div className="setup-layout">
        {/* Left: experiment info + new run */}
        <div className="setup-left">
          <section className="card">
            <h2>{exp.name || "Unnamed Experiment"}</h2>
            <div className="exp-summary">
              <strong>{exp.view}</strong> &middot;{" "}
              {exp.num_models} model{exp.num_models !== 1 ? "s" : ""}
              {exp.include_real ? " + real" : ""} &middot;{" "}
              {exp.samples_per_split}/split &middot;{" "}
              {exp.blind ? "blinded" : "not blinded"}
            </div>

            <div className="exp-models">
              <ModelTable
                models={exp.models}
                prettyNames={exp.pretty_names}
                includeReal={exp.include_real}
              />
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

            <button className="btn btn-primary" onClick={handleNewRun}>
              Start Run
            </button>
          </section>

          {/* Run history */}
          {exp.runs && exp.runs.length > 0 && (
            <section className="card">
              <h2>Runs ({exp.runs.length})</h2>
              <div className="experiment-list">
                {[...exp.runs].sort((a, b) => b.created_at.localeCompare(a.created_at)).map((run) => (
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
                        <span className="run-date">{formatShortDate(run.created_at)}</span>
                      </span>
                    </div>
                    <div className="exp-actions">
                      {run.progress > 0 && (
                        <button
                          className="btn btn-small"
                          onClick={(e) => handleRunInfo(run.id, e)}
                          title="View results"
                        >
                          &#9432;
                        </button>
                      )}
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

      {runResults && (
        <RunResultsModal
          results={runResults}
          onClose={() => setRunResults(null)}
        />
      )}

      {showSettings && (
        <div className="modal-overlay" onClick={() => setShowSettings(false)}>
          <div className="modal card" onClick={(e) => e.stopPropagation()}>
            <h3>Experiment Settings</h3>

            <label>
              Samples per split
              <NumberInput
                value={settingsSamples}
                min={1}
                max={100}
                onChange={(v) => setSettingsSamples(v)}
              />
            </label>
            <div className="settings-total">
              Total: {settingsSamples * (exp.num_models + (exp.include_real ? 1 : 0))} samples
              ({exp.num_models + (exp.include_real ? 1 : 0)} split{(exp.num_models + (exp.include_real ? 1 : 0)) !== 1 ? "s" : ""})
            </div>

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={settingsBlind}
                onChange={(e) => setSettingsBlind(e.target.checked)}
              />
              Blinded generation
            </label>

            <div className="settings-actions">
              <button className="btn btn-cancel" onClick={() => setShowSettings(false)}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={saveSettings}>
                Save
              </button>
            </div>
          </div>
        </div>
      )}
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

function formatShortDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  const mon = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  const h = String(d.getHours()).padStart(2, "0");
  const m = String(d.getMinutes()).padStart(2, "0");
  return `${mon}/${day} ${h}:${m}`;
}
