// src/App.jsx
import { useEffect, useMemo, useState } from "react";
import CitationNetworkView from "./CitationNetworkView"; // visualization only

export default function App() {
  const [rawData, setRawData] = useState(null);

  // Load your dummy dataset. Place it at: public/dummy-dataset.json
  useEffect(() => {
    fetch("/dummy-dataset.json")
      .then((r) => r.json())
      .then(setRawData)
      .catch((e) => console.error("Failed to load dataset:", e));
  }, []);

  // Normalize to { nodes, links }. If links are absent, derive from outCitations.
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

  if (!initialData) {
    return <div className="h-screen bg-gray-100 p-6 text-slate-600">Loading dataset…</div>;
  }

  return <CitationNetworkView initialData={initialData} />;
}
