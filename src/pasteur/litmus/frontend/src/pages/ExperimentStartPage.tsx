import { useState } from "react";
import { startExperiment, fetchExperiment } from "../api";
import type { ExperimentDetail } from "../api";

interface Props {
  experiment: ExperimentDetail;
  onStarted: (exp: ExperimentDetail) => void;
  onBack: () => void;
}

export default function ExperimentStartPage({
  experiment,
  onStarted,
  onBack,
}: Props) {
  const [name, setName] = useState("");
  const [tutorial, setTutorial] = useState(false);

  const handleStart = async () => {
    await startExperiment(experiment.id, name, tutorial);
    const updated = await fetchExperiment(experiment.id);
    onStarted(updated);
  };

  return (
    <div className="page start-page">
      <header className="header">
        <h1>LITMUS</h1>
        <button className="btn btn-small" onClick={onBack}>
          Back
        </button>
      </header>

      <section className="card start-card">
        <h2>Start Experiment</h2>

        <p className="exp-summary">
          <strong>{experiment.view}</strong> &middot;{" "}
          {experiment.num_models} model
          {experiment.num_models !== 1 ? "s" : ""}
          {experiment.include_real ? " + real data" : ""} &middot;{" "}
          {experiment.total_samples} samples
          {experiment.blind ? " (blinded)" : ""}
        </p>

        <label>
          Experiment name
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
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

        <button className="btn btn-primary" onClick={handleStart}>
          Start
        </button>
      </section>
    </div>
  );
}
