// src/App.jsx
import { useEffect, useMemo, useState } from "react";
import CitationUploadPage from "./CitationUploadPage";
import CitationNetworkView from "./CitationNetworkView";

export default function App() {
  const [view, setView] = useState("viz"); // "upload" | "viz"
  const [rawData, setRawData] = useState(null);

  useEffect(() => {
    fetch("/dummy-dataset.json")
      .then((r) => r.json())
      .then(setRawData)
      .catch((e) => console.error("Failed to load dataset:", e));
  }, []);

  const initialData = useMemo(() => {
    if (!rawData) return null;

    if (Array.isArray(rawData.links)) {
      return { nodes: rawData.nodes ?? [], links: rawData.links };
    }
    const nodes = (rawData.nodes ?? []).map((d) => ({ ...d }));
    const idSet = new Set(nodes.map((n) => String(n.id)));
    const links = [];
    for (const s of nodes) {
      const outs = Array.isArray(s.outCitations) ? s.outCitations : [];
      for (const t of outs) {
        const tid = String(t);
        if (idSet.has(tid)) links.push({ source: String(s.id), target: tid });
      }
    }
    return { nodes, links };
  }, [rawData]);

  if (view === "upload") return <CitationUploadPage />;

  if (!initialData) {
    return <div className="h-screen bg-gray-100 p-6 text-slate-600">Loading dataset…</div>;
  }

  return (
    <CitationNetworkView
      initialData={initialData}
      onBack={() => setView("upload")}
    />
  );
}
