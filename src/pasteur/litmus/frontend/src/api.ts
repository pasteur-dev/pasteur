/** API client for LITMUS backend */

const BASE = "/api";

/**
 * Long-poll for data changes. Blocks until version changes or ~10s timeout.
 * Returns the new version number.
 */
export async function pollChanges(knownVersion: number): Promise<number> {
  const res = await fetch(`${BASE}/poll?version=${knownVersion}`);
  const data = await res.json();
  return data.version;
}

export interface ModelVersion {
  version: string;
  name: string;
  overrides: Record<string, unknown>;
}

export interface ViewModels {
  [algorithm: string]: ModelVersion[];
}

export interface ModelRef {
  algorithm: string;
  timestamp: string;
  overrides: Record<string, unknown>;
}

export interface RunSummary {
  id: string;
  name: string;
  tutorial: boolean;
  started: boolean;
  finished: boolean;
  progress: number;
  total_samples: number;
  created_at: string;
}

export interface RunDetail extends RunSummary {
  ratings: Rating[];
}

export interface ExperimentSummary {
  id: string;
  name: string;
  view: string;
  models: ModelRef[];
  num_models: number;
  include_real: boolean;
  blind: boolean;
  samples_per_split: number;
  total_samples: number;
  num_runs: number;
  created_at: string;
  pretty_names: Record<string, string>;
}

export interface ExperimentDetail extends ExperimentSummary {
  runs: RunSummary[];
}

export interface Rating {
  entity_id: string;
  source: string;
  score: number | null;
  response_time_ms: number | null;
  skipped: boolean;
}

export interface SourceResults {
  pretty_name: string;
  count: number;
  mean: number;
  std: number;
  median: number;
  distribution: Record<number, number>;
}

export interface ResponseTimeStats {
  source: string;
  pretty_name: string;
  mean: number;
  count: number;
  times: number[];
}

export interface InterRaterSource {
  pretty_name: string;
  alpha: number;
  n_raters: number;
  n_items: number;
}

export interface InterRaterResult {
  overall: number | null;
  n_raters: number;
  per_source: InterRaterSource[];
}

export interface HumanLLMSourceDiff {
  source: string;
  pretty_name: string;
  human_mean: number;
  llm_mean: number;
  diff: number;
}

export interface HumanLLMComparison {
  per_source: HumanLLMSourceDiff[];
  human_ranking: string[];
  llm_ranking: string[];
  rank_match: boolean;
  n_sources: number;
}

export interface ExperimentResults {
  experiment_id: string;
  name: string;
  view: string;
  num_runs: number;
  total_rated: number;
  total_skipped: number;
  by_source: SourceResults[];
  llm_scores: SourceResults[];
  response_times: ResponseTimeStats[];
  inter_rater: InterRaterResult | null;
  human_llm_comparison: HumanLLMComparison | null;
}

// --- Views ---

export async function fetchViews(): Promise<string[]> {
  const res = await fetch(`${BASE}/views`);
  return res.json();
}

export async function fetchModels(view: string): Promise<ViewModels> {
  const res = await fetch(`${BASE}/views/${view}/models`);
  return res.json();
}

// --- Experiments ---

export async function fetchExperiments(): Promise<ExperimentSummary[]> {
  const res = await fetch(`${BASE}/experiments`);
  return res.json();
}

export async function fetchExperiment(id: string): Promise<ExperimentDetail> {
  const res = await fetch(`${BASE}/experiments/${id}`);
  return res.json();
}

export async function createExperiment(data: {
  name: string;
  view: string;
  models: ModelRef[];
  include_real: boolean;
  blind: boolean;
  samples_per_split: number;
}): Promise<ExperimentDetail> {
  const res = await fetch(`${BASE}/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function deleteExperiment(id: string): Promise<void> {
  await fetch(`${BASE}/experiments/${id}`, { method: "DELETE" });
}

export async function updateExperiment(
  id: string,
  updates: { samples_per_split?: number; blind?: boolean }
): Promise<ExperimentDetail> {
  const res = await fetch(`${BASE}/experiments/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  return res.json();
}

// --- Runs ---

export async function createRun(
  experimentId: string,
  name: string,
  tutorial: boolean
): Promise<RunDetail> {
  const res = await fetch(`${BASE}/experiments/${experimentId}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, tutorial }),
  });
  return res.json();
}

export async function deleteRun(
  experimentId: string,
  runId: string
): Promise<void> {
  await fetch(`${BASE}/experiments/${experimentId}/runs/${runId}`, {
    method: "DELETE",
  });
}

export async function rateEntity(
  experimentId: string,
  runId: string,
  entityId: string,
  source: string,
  score: number,
  responseTimeMs: number
): Promise<{ ok: boolean; progress: number; total: number }> {
  const res = await fetch(
    `${BASE}/experiments/${experimentId}/runs/${runId}/rate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        entity_id: entityId,
        source,
        score,
        response_time_ms: responseTimeMs,
      }),
    }
  );
  return res.json();
}

export async function skipEntity(
  experimentId: string,
  runId: string,
  entityId: string,
  source: string
): Promise<void> {
  await fetch(
    `${BASE}/experiments/${experimentId}/runs/${runId}/skip`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entity_id: entityId, source }),
    }
  );
}

export async function undoRating(
  experimentId: string,
  runId: string
): Promise<{ ok: boolean; progress: number }> {
  const res = await fetch(
    `${BASE}/experiments/${experimentId}/runs/${runId}/undo`,
    { method: "POST" }
  );
  return res.json();
}

export async function endRun(
  experimentId: string,
  runId: string
): Promise<void> {
  await fetch(
    `${BASE}/experiments/${experimentId}/runs/${runId}/end`,
    { method: "POST" }
  );
}

export async function resumeRun(
  experimentId: string,
  runId: string
): Promise<void> {
  await fetch(
    `${BASE}/experiments/${experimentId}/runs/${runId}/resume`,
    { method: "POST" }
  );
}

export async function fetchResults(id: string): Promise<ExperimentResults> {
  const res = await fetch(`${BASE}/experiments/${id}/results`);
  return res.json();
}

export interface RunResults {
  run_id: string;
  name: string;
  total_rated: number;
  total_skipped: number;
  by_source: SourceResults[];
  response_times: ResponseTimeStats[];
}

export async function fetchRunResults(
  experimentId: string,
  runId: string
): Promise<RunResults> {
  const res = await fetch(
    `${BASE}/experiments/${experimentId}/runs/${runId}/results`
  );
  return res.json();
}
