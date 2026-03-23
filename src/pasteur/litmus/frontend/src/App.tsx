import { useState } from "react";
import type { ExperimentDetail } from "./api";
import SetupPage from "./pages/SetupPage";
import ExperimentStartPage from "./pages/ExperimentStartPage";
import EvaluationPage from "./pages/EvaluationPage";

type Page =
  | { type: "setup" }
  | { type: "start"; experiment: ExperimentDetail }
  | { type: "evaluate"; experiment: ExperimentDetail };

export default function App() {
  const [page, setPage] = useState<Page>({ type: "setup" });

  switch (page.type) {
    case "setup":
      return (
        <SetupPage
          onExperimentCreated={(exp) =>
            setPage({ type: "start", experiment: exp })
          }
          onResumeExperiment={(exp) =>
            setPage({ type: "evaluate", experiment: exp })
          }
        />
      );
    case "start":
      return (
        <ExperimentStartPage
          experiment={page.experiment}
          onStarted={(exp) => setPage({ type: "evaluate", experiment: exp })}
          onBack={() => setPage({ type: "setup" })}
        />
      );
    case "evaluate":
      return (
        <EvaluationPage
          experiment={page.experiment}
          onFinished={() => setPage({ type: "setup" })}
        />
      );
  }
}
