import { useCallback, useEffect, useState } from "react";
import { fetchExperiment } from "./api";
import type { ExperimentDetail } from "./api";
import SetupPage from "./pages/SetupPage";
import ExperimentPage from "./pages/ExperimentPage";
import EvaluationPage from "./pages/EvaluationPage";

type Route =
  | { type: "setup" }
  | { type: "experiment"; experimentId: string }
  | { type: "evaluate"; experimentId: string; runId: string };

function parseHash(): Route {
  const hash = window.location.hash.replace(/^#\/?/, "");
  const parts = hash.split("/");

  if (parts[0] === "experiment" && parts[1]) {
    if (parts[2] === "run" && parts[3]) {
      return { type: "evaluate", experimentId: parts[1], runId: parts[3] };
    }
    return { type: "experiment", experimentId: parts[1] };
  }
  return { type: "setup" };
}

function navigate(route: Route) {
  switch (route.type) {
    case "setup":
      window.location.hash = "/";
      break;
    case "experiment":
      window.location.hash = `/experiment/${route.experimentId}`;
      break;
    case "evaluate":
      window.location.hash = `/experiment/${route.experimentId}/run/${route.runId}`;
      break;
  }
}

export default function App() {
  const [route, setRoute] = useState<Route>(parseHash);
  const [experiment, setExperiment] = useState<ExperimentDetail | null>(null);
  const [loading, setLoading] = useState(false);

  // Listen to hash changes (back/forward buttons)
  useEffect(() => {
    const onHashChange = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  // Load experiment data when route needs it
  useEffect(() => {
    if (route.type === "setup") {
      setExperiment(null);
      return;
    }

    const eid = route.experimentId;
    if (experiment?.id === eid) return;

    setLoading(true);
    fetchExperiment(eid)
      .then(setExperiment)
      .catch(() => {
        navigate({ type: "setup" });
      })
      .finally(() => setLoading(false));
  }, [route, experiment?.id]);

  const goToSetup = useCallback(() => {
    navigate({ type: "setup" });
  }, []);

  const goToExperiment = useCallback((exp: ExperimentDetail) => {
    setExperiment(exp);
    navigate({ type: "experiment", experimentId: exp.id });
  }, []);

  const goToRun = useCallback(
    (exp: ExperimentDetail, runId: string) => {
      setExperiment(exp);
      navigate({ type: "evaluate", experimentId: exp.id, runId });
    },
    []
  );

  if (loading) {
    return (
      <div className="page">
        <div className="entity-loading">Loading...</div>
      </div>
    );
  }

  switch (route.type) {
    case "setup":
      return <SetupPage onSelectExperiment={goToExperiment} />;

    case "experiment":
      if (!experiment) return null;
      return (
        <ExperimentPage
          experiment={experiment}
          onStartRun={(exp, run) => goToRun(exp, run.id)}
          onResumeRun={(exp, run) => goToRun(exp, run.id)}
          onBack={goToSetup}
        />
      );

    case "evaluate":
      if (!experiment) return null;
      return (
        <EvaluationPage
          experiment={experiment}
          runId={route.runId}
          onFinished={() => goToExperiment(experiment)}
        />
      );
  }
}
