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
import { ModelTable } from "../components/ModelLabel";
import NumberInput from "../components/NumberInput";

interface Props {
  onSelectExperiment: (exp: ExperimentDetail) => void;
}

interface SelectedModel {
  algorithm: string;
  version: string;
}

export default function SetupPage({ onSelectExperiment }: Props) {
  const [views, setViews] = useState<string[]>([]);
  const [selectedView, setSelectedView] = useState("");
  const [models, setModels] = useState<ViewModels>({});
  const [selectedModels, setSelectedModels] = useState<SelectedModel[]>([]);
  const [expandedAlgs, setExpandedAlgs] = useState<Set<string>>(new Set());
  const [includeReal, setIncludeReal] = useState(true);
  const [blind, setBlind] = useState(true);
  const [samplesPerSplit, setSamplesPerSplit] = useState(20);
  const [expName, setExpName] = useState("");
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);

  useEffect(() => {
    fetchViews().then(setViews);
    fetchExperiments().then(setExperiments);
  }, []);

  useEffect(() => {
    if (selectedView) {
      fetchModels(selectedView).then(setModels);
      setSelectedModels([]);
    }
  }, [selectedView]);

  const numSplits = selectedModels.length + (includeReal ? 1 : 0);
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
    const modelRefs: ModelRef[] = selectedModels.map((m) => {
      // Find the overrides from the discovered model data
      const versions = models[m.algorithm] || [];
      const found = versions.find((v) => v.version === m.version);
      return {
        algorithm: m.algorithm,
        timestamp: m.version,
        overrides: found?.overrides || {},
      };
    });
    const exp = await createExperiment({
      name: expName || `${selectedView} experiment (${randomTag()})`,
      view: selectedView,
      models: modelRefs,
      include_real: includeReal,
      blind,
      samples_per_split: samplesPerSplit,
    });
    onSelectExperiment(exp);
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await deleteExperiment(id);
    setExperiments((prev) => prev.filter((exp) => exp.id !== id));
  };

  const handleSelect = async (id: string) => {
    const exp = await fetchExperiment(id);
    onSelectExperiment(exp);
  };

  return (
    <div className="page setup-page">
      <header className="header">
        <a href="#/" className="header-brand">
          <img src="/logo.svg" alt="Pasteur" className="header-logo" />
          <h1>LITMUS</h1>
        </a>
      </header>

      <div className="setup-layout">
        {/* Left: create new experiment */}
        <div className="setup-left">
          <section className="card">
            <h2>New Experiment</h2>

            <label>
              Experiment name
              <input
                type="text"
                value={expName}
                onChange={(e) => setExpName(e.target.value)}
                placeholder="e.g. MIMIC realism study"
              />
            </label>

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

            {selectedView && (
              <div className="model-list">
                <h3>Models</h3>
                {Object.entries(models).map(([alg, versions]) => (
                  <div key={alg} className="model-group">
                    <h4>{alg}</h4>
                    {versions
                      .slice(0, expandedAlgs.has(alg) ? undefined : 5)
                      .map((v) => {
                        const isSelected = selectedModels.some(
                          (m) =>
                            m.algorithm === alg && m.version === v.version
                        );
                        return (
                          <label
                            key={v.version}
                            className={`model-item ${isSelected ? "selected" : ""}`}
                          >
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => toggleModel(alg, v.version)}
                            />
                            {v.name ? (
                              <span className="model-item-name">
                                <span className="model-item-overrides">{v.name}</span>
                                <span className="model-item-ts">{formatTimestamp(v.version)}</span>
                              </span>
                            ) : (
                              <span className="model-item-ts">{formatTimestamp(v.version)}</span>
                            )}
                          </label>
                        );
                      })}
                    {versions.length > 5 && !expandedAlgs.has(alg) && (
                      <button
                        className="btn-link"
                        onClick={() =>
                          setExpandedAlgs((s) => new Set([...s, alg]))
                        }
                      >
                        Show {versions.length - 5} more...
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div className="options">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={blind}
                  onChange={(e) => setBlind(e.target.checked)}
                />
                Blinded (hide source during evaluation)
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={includeReal}
                  onChange={(e) => setIncludeReal(e.target.checked)}
                />
                Include real data split
              </label>
            </div>

            <div className="sample-counts">
              <label>
                Samples per split
                <NumberInput
                  value={samplesPerSplit}
                  min={1}
                  max={100}
                  onChange={(v) => setSamplesPerSplit(v)}
                />
              </label>
              <label>
                Total samples ({numSplits} split
                {numSplits !== 1 ? "s" : ""})
                <NumberInput
                  value={totalSamples}
                  min={numSplits}
                  onChange={(total) =>
                    setSamplesPerSplit(
                      Math.max(1, Math.round(total / Math.max(1, numSplits)))
                    )
                  }
                />
              </label>
            </div>

            <button
              className="btn btn-primary"
              disabled={selectedModels.length === 0}
              onClick={handleCreate}
            >
              Create Experiment
            </button>
          </section>
        </div>

        {/* Right: existing experiments */}
        <div className="setup-right setup-right-narrow">
          {experiments.length > 0 && (
            <section className="card">
              <h2>Experiments</h2>
              <div className="experiment-list">
                {experiments.map((exp) => (
                  <div
                    key={exp.id}
                    className="experiment-item experiment-item-rich"
                    onClick={() => handleSelect(exp.id)}
                  >
                    <div className="exp-item-top">
                      <div className="exp-info">
                        <strong>{exp.name || "Unnamed"}</strong>
                        <span className="exp-meta">
                          {exp.view} · {exp.samples_per_split}/split ·{" "}
                          {exp.total_samples} total · {exp.num_runs} run
                          {exp.num_runs !== 1 ? "s" : ""} ·{" "}
                          {exp.blind ? "blinded" : "not blinded"}
                        </span>
                      </div>
                      <div className="exp-actions">
                        <button
                          className="btn btn-small btn-danger"
                          onClick={(e) => handleDelete(exp.id, e)}
                          title="Delete"
                        >
                          &#128465;
                        </button>
                      </div>
                    </div>
                    <div className="exp-item-models">
                      <ModelTable
                        models={exp.models}
                        prettyNames={exp.pretty_names}
                        includeReal={exp.include_real}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
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

const TAGS = [
  "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
  "iota", "kappa", "lambda", "sigma", "omega", "nova", "flux", "pulse",
  "spark", "drift", "nexus", "prism", "helix", "vortex", "cipher", "quasar",
];

function randomTag(): string {
  const tag = TAGS[Math.floor(Math.random() * TAGS.length)];
  const num = Math.floor(Math.random() * 100);
  return `${tag}-${num}`;
}
