// CitationNetworkView.jsx
import React, { useMemo, useState } from "react";
import { ArrowLeft, CheckCircle2, RotateCcw } from "lucide-react";
import GraphPanel from "./GraphPanel";
import DetailsPanel from "./DetailsPanel";

export default function CitationNetworkView({ initialData }) {
  const [selectedNode, setSelectedNode] = useState(null);
  const [reloadKey, setReloadKey] = useState(0); // remount graph on reload

  const data = useMemo(() => {
    const nodes = (initialData.nodes ?? []).map((d) => ({
      ...d,
      id: String(d.id),
      inCitations: Array.isArray(d.inCitations) ? [...d.inCitations] : [],
    }));
    const idMap = new Map(nodes.map((n) => [n.id, n]));
    const links = (initialData.links ?? [])
      .map((l) => ({ source: String(l.source), target: String(l.target) }))
      .filter((l) => idMap.has(l.source) && idMap.has(l.target));

    // compute inCitations from links
    for (const n of nodes) n.inCitations = [];
    for (const l of links) idMap.get(l.target)?.inCitations.push(l.source);

    // degree (incident edges) for coloring
    const degree = new Map(nodes.map((n) => [n.id, 0]));
    for (const l of links) {
      degree.set(l.source, (degree.get(l.source) || 0) + 1);
      degree.set(l.target, (degree.get(l.target) || 0) + 1);
    }
    const maxDeg = Math.max(0, ...degree.values());
    return { nodes, links, degree, maxDeg, idMap };
  }, [initialData]);

  // programmatic jump from details lists
  const jumpToId = (id) => {
    const n = data.idMap.get(String(id));
    if (n) setSelectedNode(n);
  };

  return (
    <div className="h-screen bg-gray-100 text-gray-900 font-sans p-6">
      {/* Header — Rowan-like */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            className="inline-flex items-center gap-1 px-2 py-1 rounded-md border bg-white hover:bg-gray-50"
            onClick={() => console.log("TODO: navigate to upload")}
            title="Back to upload"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm">Back</span>
          </button>
          <div className="text-xl font-semibold tracking-tight flex items-center gap-2">
            <span>MVP Citation Explorer (dummy title)</span>
            <CheckCircle2 className="w-4 h-4 text-green-500" title="Extracted ✓" />
          </div>
        </div>

        <button
          className="inline-flex items-center gap-2 px-3 py-1 rounded-md border bg-white hover:bg-gray-50"
          onClick={() => setReloadKey((k) => k + 1)}
          title="Reload graph layout"
        >
          <RotateCcw className="w-4 h-4" />
          <span className="text-sm">Reload</span>
        </button>
      </div>

      <div className="h-[calc(100vh-7.5rem)] grid grid-cols-[1fr_360px] gap-6">
        <div className="bg-white rounded-lg shadow-sm h-full">
          <GraphPanel
            key={reloadKey}
            data={data}
            selectedId={selectedNode?.id || null} // center on selection
            onSelect={setSelectedNode}
            className="h-full"
          />
        </div>

        <DetailsPanel
          node={selectedNode}
          nodes={data.nodes}
          onJumpToId={jumpToId}    // make refs/citations clickable
          className="h-full"
        />
      </div>
    </div>
  );
}
