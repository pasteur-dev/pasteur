import { useState } from "react";
import type { ExperimentDetail } from "./api";
import SetupPage from "./pages/SetupPage";
import ExperimentPage from "./pages/ExperimentPage";
import EvaluationPage from "./pages/EvaluationPage";

type Page =
  | { type: "setup" }
  | { type: "experiment"; experiment: ExperimentDetail }
  | { type: "evaluate"; experiment: ExperimentDetail; runId: string };

export default function App() {
  const [page, setPage] = useState<Page>({ type: "setup" });

  switch (page.type) {
    case "setup":
      return (
        <SetupPage
          onSelectExperiment={(exp) =>
            setPage({ type: "experiment", experiment: exp })
          }
        />
      );
    case "experiment":
      return (
        <ExperimentPage
          experiment={page.experiment}
          onStartRun={(exp, run) =>
            setPage({ type: "evaluate", experiment: exp, runId: run.id })
          }
          onResumeRun={(exp, run) =>
            setPage({ type: "evaluate", experiment: exp, runId: run.id })
          }
          onBack={() => setPage({ type: "setup" })}
        />
      );
    case "evaluate":
      return (
        <EvaluationPage
          experiment={page.experiment}
          runId={page.runId}
          onFinished={() =>
            setPage({ type: "experiment", experiment: page.experiment })
          }
        />
      );
  }
}
