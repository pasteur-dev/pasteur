import { useEffect, useState } from "react";
import {
  fetchViews,
  fetchModels,
  fetchExperiments,
  createExperiment,
  deleteExperiment,
  fetchExperiment,
} from "../api";
import type {
  ViewModels,
  ExperimentDetail,
  ExperimentSummary,
  ModelRef,
} from "../api";
import ResultsPanel from "./ResultsPanel";

interface Props {
  onExperimentCreated: (exp: ExperimentDetail) => void;
  onResumeExperiment: (exp: ExperimentDetail) => void;
}

interface SelectedModel {
  algorithm: string;
  version: string;
}

export default function SetupPage({
  onExperimentCreated,
  onResumeExperiment,
}: Props) {
  const [views, setViews] = useState<string[]>([]);
  const [selectedView, setSelectedView] = useState("");
  const [models, setModels] = useState<ViewModels>({});
  const [selectedModels, setSelectedModels] = useState<SelectedModel[]>([]);
  const [expandedAlgs, setExpandedAlgs] = useState<Set<string>>(new Set());
  const [includeReal, setIncludeReal] = useState(true);
  const [blind, setBlind] = useState(true);
  const [samplesPerSplit, setSamplesPerSplit] = useState(20);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);

  // Load views on mount
  useEffect(() => {
    fetchViews().then(setViews);
    fetchExperiments().then(setExperiments);
  }, []);

  // Load models when view changes
  useEffect(() => {
    if (selectedView) {
      fetchModels(selectedView).then(setModels);
      setSelectedModels([]);
    }
  }, [selectedView]);

  const numSplits =
    selectedModels.length + (includeReal ? 1 : 0);
  const totalSamples = numSplits * samplesPerSplit;

  const toggleModel = (alg: string, version: string) => {
    setSelectedModels((prev) => {
      const exists = prev.some(
        (m) => m.algorithm === alg && m.version === version
      );
      if (exists) {
        return prev.filter(
          (m) => !(m.algorithm === alg && m.version === version)
        );
      }
      return [...prev, { algorithm: alg, version }];
    });
  };

  const handleCreate = async () => {
    const modelRefs: ModelRef[] = selectedModels.map((m) => ({
      algorithm: m.algorithm,
      timestamp: m.version,
      overrides: {},
    }));
    const exp = await createExperiment({
      view: selectedView,
      models: modelRefs,
      include_real: includeReal,
      blind,
      samples_per_split: samplesPerSplit,
    });
    onExperimentCreated(exp);
  };

  const handleDelete = async (id: string) => {
    await deleteExperiment(id);
    setExperiments((prev) => prev.filter((e) => e.id !== id));
    if (selectedExpId === id) setSelectedExpId(null);
  };

  const handleResume = async (id: string) => {
    const exp = await fetchExperiment(id);
    onResumeExperiment(exp);
  };

  return (
    <div className="page setup-page">
      <header className="header">
        <a href="/" className="header-brand">
          <img src="/logo.svg" alt="Pasteur" className="header-logo" />
          <h1>LITMUS</h1>
        </a>
      </header>

      <div className="setup-layout">
        {/* Left panel: New experiment + history */}
        <div className="setup-left">
          <section className="card">
            <h2>New Experiment</h2>

            {/* View selector */}
            <label>
              View
              <select
                value={selectedView}
                onChange={(e) => setSelectedView(e.target.value)}
              >
                <option value="">Select a view...</option>
                {views.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>

            {/* Model list */}
            {selectedView && (
              <div className="model-list">
                <h3>Models</h3>
                {Object.entries(models).map(([alg, versions]) => (
                  <div key={alg} className="model-group">
                    <h4>{alg}</h4>
                    {versions
                      .slice(
                        0,
                        expandedAlgs.has(alg) ? undefined : 5
                      )
                      .map((v) => {
                        const isSelected = selectedModels.some(
                          (m) =>
                            m.algorithm === alg &&
                            m.version === v.version
                        );
                        return (
                          <label
                            key={v.version}
                            className={`model-item ${isSelected ? "selected" : ""}`}
                          >
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() =>
                                toggleModel(alg, v.version)
                              }
                            />
                            <span className="version">
                              {formatTimestamp(v.version)}
                            </span>
                          </label>
                        );
                      })}
                    {versions.length > 5 &&
                      !expandedAlgs.has(alg) && (
                        <button
                          className="btn-link"
                          onClick={() =>
                            setExpandedAlgs(
                              (s) => new Set([...s, alg])
                            )
                          }
                        >
                          Show {versions.length - 5} more...
                        </button>
                      )}
                  </div>
                ))}
              </div>
            )}

            {/* Options */}
            <div className="options">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={blind}
                  onChange={(e) => setBlind(e.target.checked)}
                />
                Enable blind generation
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={includeReal}
                  onChange={(e) => setIncludeReal(e.target.checked)}
                />
                Include real data
              </label>
            </div>

            {/* Sample counts */}
            <div className="sample-counts">
              <label>
                Samples per split
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={samplesPerSplit}
                  onChange={(e) =>
                    setSamplesPerSplit(
                      Math.max(1, parseInt(e.target.value) || 1)
                    )
                  }
                />
              </label>
              <label>
                Total samples
                <input
                  type="number"
                  min={numSplits}
                  value={totalSamples}
                  onChange={(e) => {
                    const total = parseInt(e.target.value) || numSplits;
                    setSamplesPerSplit(
                      Math.max(
                        1,
                        Math.round(total / Math.max(1, numSplits))
                      )
                    );
                  }}
                />
              </label>
              <span className="split-info">
                {numSplits} split{numSplits !== 1 ? "s" : ""}
              </span>
            </div>

            <button
              className="btn btn-primary"
              disabled={selectedModels.length === 0}
              onClick={handleCreate}
            >
              Next
            </button>
          </section>

          {/* Historical experiments */}
          {experiments.length > 0 && (
            <section className="card">
              <h2>Experiments</h2>
              <div className="experiment-list">
                {experiments.map((exp) => (
                  <div
                    key={exp.id}
                    className={`experiment-item ${selectedExpId === exp.id ? "selected" : ""}`}
                    onClick={() => setSelectedExpId(exp.id)}
                  >
                    <div className="exp-info">
                      <strong>
                        {exp.name || "Unnamed"}
                      </strong>
                      <span>
                        {exp.view} &middot;{" "}
                        {exp.progress}/{exp.total_samples}
                        {exp.finished ? " (done)" : ""}
                      </span>
                    </div>
                    <div className="exp-actions">
                      {exp.started && !exp.finished && (
                        <button
                          className="btn btn-small"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleResume(exp.id);
                          }}
                        >
                          Resume
                        </button>
                      )}
                      <button
                        className="btn btn-small btn-danger"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(exp.id);
                        }}
                        title="Delete"
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

        {/* Right panel: Results dashboard */}
        <div className="setup-right">
          {selectedExpId && <ResultsPanel experimentId={selectedExpId} />}
        </div>
      </div>
    </div>
  );
}

function formatTimestamp(ts: string): string {
  // Convert "2026-01-22T09.23.40.216Z" to readable format
  return ts
    .replace(/T/, " ")
    .replace(/\.\d+Z$/, "")
    .replace(/\./g, ":");
}
