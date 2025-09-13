// CitationNetworkView.jsx
import React, { useMemo, useState } from "react";
import { ArrowLeft, CheckCircle2, RotateCcw, Search, Info } from "lucide-react";
import GraphPanel from "./GraphPanel";
import DetailsPanel from "./DetailsPanel";

export default function CitationNetworkView({ initialData, onBack }) {
  const [selectedNode, setSelectedNode] = useState(null);
  const [fitSignal, setFitSignal] = useState(0);       // animated fit (Focus)
  const [recenterSignal, setRecenterSignal] = useState(0); // instant recenter (Refresh)

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

    for (const n of nodes) n.inCitations = [];
    for (const l of links) idMap.get(l.target)?.inCitations.push(l.source);

    const degree = new Map(nodes.map((n) => [n.id, 0]));
    for (const l of links) {
      degree.set(l.source, (degree.get(l.source) || 0) + 1);
      degree.set(l.target, (degree.get(l.target) || 0) + 1);
    }
    const maxDeg = Math.max(0, ...degree.values());
    return { nodes, links, degree, maxDeg, idMap };
  }, [initialData]);

  const jumpToId = (id) => {
    const n = data.idMap.get(String(id));
    if (n) setSelectedNode(n);
  };

  return (
    <div className="h-screen bg-gray-100 text-gray-900 font-sans p-6">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-gray-200 bg-white hover:bg-gray-50"
            onClick={onBack}
            title="Back to upload"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm">Back</span>
          </button>
          <div className="text-xl font-semibold tracking-tight flex items-center gap-2">
            <span className="leading-tight">MVP Citation Explorer (dummy title)</span>
            <CheckCircle2 className="w-4 h-4 text-green-500" title="Extracted ✓" />
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Focus (animated fit all) */}
          <button
            className="inline-flex items-center justify-center w-9 h-9 rounded-md border border-gray-200 bg-gray-50 text-gray-700 hover:bg-gray-100"
            onClick={() => setFitSignal((s) => s + 1)}
            title="Focus (fit view)"
          >
            <Search className="w-4 h-4" />
          </button>
          {/* Refresh (instant recenter, keep zoom) */}
          <button
            className="inline-flex items-center justify-center w-9 h-9 rounded-md border border-gray-200 bg-gray-50 text-gray-700 hover:bg-gray-100"
            onClick={() => setRecenterSignal((s) => s + 1)}
            title="Refresh view"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="h-[calc(100vh-9.5rem)] grid grid-cols-[1fr_360px] gap-6">
        <div className="bg-white rounded-lg shadow-sm h-full">
          <GraphPanel
            data={data}
            onSelect={setSelectedNode}
            fitSignal={fitSignal}           // animated fit
            recenterSignal={recenterSignal} // instant recenter
            className="h-full"
          />
        </div>

        <DetailsPanel
          node={selectedNode}
          nodes={data.nodes}
          onJumpToId={jumpToId}
          className="h-full"
        />
      </div>

      <div className="mt-4 w-full rounded-md border border-indigo-500 bg-indigo-50 text-indigo-600 px-4 py-2 flex items-center gap-2">
        <Info className="w-4 h-4" />
        <span className="text-sm">information</span>
      </div>
    </div>
  );
}
