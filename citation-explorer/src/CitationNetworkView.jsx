// CitationNetworkView.jsx
import React, { useMemo, useState } from "react";
import GraphPanel from "./GraphPanel";
import DetailsPanel from "./DetailsPanel";

export default function CitationNetworkView({ initialData }) {
  const [selectedNode, setSelectedNode] = useState(null);

  const data = useMemo(() => {
    // clone nodes; normalize ids to strings
    const nodes = (initialData.nodes ?? []).map((d) => ({
      ...d,
      id: String(d.id),
      inCitations: Array.isArray(d.inCitations) ? [...d.inCitations] : [],
    }));

    const idMap = new Map(nodes.map((n) => [n.id, n]));
    // build links (assume provided); if not, already created upstream in App.jsx
    const links = (initialData.links ?? []).map((l) => ({
      source: String(l.source),
      target: String(l.target),
    })).filter((l) => idMap.has(l.source) && idMap.has(l.target));

    // compute inCitations (papers that cite this node) from links
    // link is source -> target (source cites target), so target gets source in its inCitations
    for (const n of nodes) n.inCitations = [];
    for (const l of links) {
      idMap.get(l.target)?.inCitations.push(l.source);
    }

    // degree for coloring (incident edges count)
    const degree = new Map(nodes.map((n) => [n.id, 0]));
    for (const l of links) {
      degree.set(l.source, (degree.get(l.source) || 0) + 1);
      degree.set(l.target, (degree.get(l.target) || 0) + 1);
    }
    const maxDeg = Math.max(0, ...degree.values());

    return { nodes, links, degree, maxDeg };
  }, [initialData]);

  return (
    <div className="h-screen bg-gray-100 text-gray-900 font-sans p-6">
      <div className="h-full grid grid-cols-[1fr_360px] gap-6">
        {/* Left: Graph */}
        <div className="bg-white rounded-lg shadow-sm h-full">
          <GraphPanel
            data={data}
            onSelect={setSelectedNode}
            className="h-full"
          />
        </div>

        {/* Right: Polished details panel (no scroll; we cap lists) */}
        <DetailsPanel
          node={selectedNode}
          nodes={data.nodes}
          className="h-full"
        />
      </div>
    </div>
  );
}
