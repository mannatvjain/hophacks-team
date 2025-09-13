import { useState } from "react";
import CitationUploadPage from "./CitationUploadPage";
import CitationNetworkView from "./CitationNetworkView";
import { fetchGraphForDOI } from "./api";

export default function App() {
  const [view, setView] = useState("upload"); // "upload" | "viz"
  const [graphData, setGraphData] = useState(null);

  async function handleSubmitDOI(doi) {
    const data = await fetchGraphForDOI(doi); // { nodes, links }
    setGraphData(data);
    setView("viz");
  }

  if (view === "upload") {
    return <CitationUploadPage onSubmitDOI={handleSubmitDOI} />;
  }

  return (
    <CitationNetworkView
      initialData={graphData}
      onBack={() => setView("upload")}
    />
  );
}
