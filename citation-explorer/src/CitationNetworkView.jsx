// CitationNetworkView.jsx
import React, { useMemo, useState } from "react";
import { ArrowLeft, CheckCircle2, RotateCcw, Search, Info, Mail } from "lucide-react";
import GraphPanel from "./GraphPanel";
import DetailsPanel from "./DetailsPanel";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function CitationNetworkView({ initialData, onBack, doi }) {
  // selection + camera
  const [selectedNode, setSelectedNode] = useState(null);
  const [fitSignal, setFitSignal] = useState(0);         // animated fit (Focus)
  const [recenterSignal, setRecenterSignal] = useState(0); // instant recenter (Refresh)

  // reading list
  const [readingIds, setReadingIds] = useState(new Set());

  // BFS depth (controls backend subgraph size)
  const [depth, setDepth] = useState(3);

  // live graph data (starts from server-provided initialData)
  const [graphData, setGraphData] = useState(initialData || { nodes: [], links: [], shortest_distance: [] });

  // ---- fetcher for depth changes ----
  async function fetchGraph(nextDepth) {
    const body = { doi: String(doi || ""), depth: Number(nextDepth) };
    const res = await fetch(`${API_URL}/api/graph`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const json = await res.json();
    setGraphData(json);
    setSelectedNode(null);
    setFitSignal((s) => s + 1); // smooth refit after redraw
  }

  const data = useMemo(() => {
    const nodes = (graphData.nodes ?? []).map((d) => ({
      ...d,
      id: String(d.id),
      inCitations: Array.isArray(d.inCitations) ? [...d.inCitations] : [],
    }));
    const idMap = new Map(nodes.map((n) => [n.id, n]));

    const links = (graphData.links ?? [])
      .map((l) => ({ source: String(l.source), target: String(l.target) }))
      .filter((l) => idMap.has(l.source) && idMap.has(l.target));

    // build in-citations
    for (const n of nodes) n.inCitations = [];
    for (const l of links) idMap.get(l.target)?.inCitations.push(l.source);

    // scores + TOP 10 ids by score (ties by id for stability)
    const score = new Map(nodes.map((n) => [n.id, Number.isFinite(n.score) ? Number(n.score) : 0]));
    const sorted = [...nodes].sort(
      (a, b) => (Number(b.score) || 0) - (Number(a.score) || 0) || String(a.id).localeCompare(String(b.id))
    );
    const goldId = sorted.length ? String(sorted[0].id) : null;
    const top10Ids = new Set(sorted.slice(1, 11).map((n) => String(n.id)));
    const goldTitle = goldId ? (sorted[0].title || String(sorted[0].id)) : "Untitled";

    // ---- shortest paths (gold → each orange), gather *directed* edges to highlight ----
    const linkKeys = new Set(links.map((l) => `${l.source}->${l.target}`)); // for direction
    const adj = new Map(nodes.map((n) => [n.id, new Set()]));               // undirected for BFS
    for (const l of links) {
      adj.get(l.source)?.add(l.target);
      adj.get(l.target)?.add(l.source);
    }

    // BFS from gold
    const parents = new Map();
    const visited = new Set();
    const q = [];
    if (goldId) {
      parents.set(goldId, null);
      visited.add(goldId);
      q.push(goldId);
    }
    while (q.length) {
      const u = q.shift();
      for (const v of adj.get(u) ?? []) {
        if (!visited.has(v)) {
          visited.add(v);
          parents.set(v, u);
          q.push(v);
        }
      }
    }

    // Reconstruct edge keys for all gold→orange paths, preserving real arrow direction
    const pathEdgeKeys = new Set();
    for (const tid of top10Ids) {
      if (tid === goldId) continue;
      if (!parents.has(tid)) continue;
      let v = tid;
      while (parents.get(v) != null) {
        const u = parents.get(v);
        const k1 = `${u}->${v}`, k2 = `${v}->${u}`;
        if (linkKeys.has(k1)) pathEdgeKeys.add(k1);
        else if (linkKeys.has(k2)) pathEdgeKeys.add(k2);
        v = u;
      }
    }

    return {
      nodes,
      links,
      score,
      goldId,
      goldTitle,
      top10Ids,
      idMap,
      // passthrough from backend (unused if pathEdgeKeys present)
      shortest_distance: (graphData.shortest_distance ?? []).map(String),
      // NEW: edges to highlight in orange
      pathEdgeKeys: [...pathEdgeKeys],
    };
  }, [graphData]);

  const jumpToId = (id) => {
    const n = data.idMap.get(String(id));
    if (n) setSelectedNode(n);
  };

  const toggleReading = (id) => {
    setReadingIds((prev) => {
      const next = new Set(prev);
      const key = String(id);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const readingTitles = [...readingIds].map((id) => data.idMap.get(id)?.title || id);

  const handleExport = () => {
    const subject = encodeURIComponent("Recommended Reading");
    const originalPaper = data.goldTitle || "the original paper";
    let lines = [];
    lines.push(`Recommended reading based on ${originalPaper}`);
    lines.push("");
    lines = lines.concat(readingTitles.map((t, i) => `${i + 1}. ${t}`));
    const bodyPlain = lines.join("\r\n");
    const body = encodeURIComponent(bodyPlain);
    const to = "your.email@example.com";
    window.location.href = `mailto:${to}?subject=${subject}&body=${body}`;
  };

  // handle depth control clicks from GraphPanel
  const handleDepthChange = (nextDepth) => {
    setDepth(nextDepth);
    fetchGraph(nextDepth);
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
            <span className="leading-tight">{data.goldTitle}</span>
            <CheckCircle2 className="w-4 h-4 text-green-500" title="Extracted ✓" />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            className="inline-flex items-center justify-center w-9 h-9 rounded-md border border-gray-200 bg-gray-50 text-gray-700 hover:bg-gray-100"
            onClick={() => setFitSignal((s) => s + 1)}
            title="Focus (fit view)"
          >
            <Search className="w-4 h-4" />
          </button>
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
            onToggleReading={toggleReading}
            readingIds={readingIds}
            fitSignal={fitSignal}
            recenterSignal={recenterSignal}
            mode="full"
            depth={depth}
            onDepthChange={handleDepthChange}
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

      {/* Reading strip */}
      <ReadingStrip titles={readingTitles} onExport={handleExport} />
    </div>
  );
}

/* Thin indigo reading strip; scrollbar hidden but still scrollable; export at end */
function ReadingStrip({ titles, onExport }) {
  return (
    <>
      <style>{`
        #reading-strip { scrollbar-width: none; -ms-overflow-style: none; }
        #reading-strip::-webkit-scrollbar { display: none; }
      `}</style>
      <div
        id="reading-strip"
        className="mt-3 w-full rounded-md border border-indigo-500 bg-indigo-50 text-indigo-600 px-3 py-1.5 flex items-center gap-2 overflow-x-auto whitespace-nowrap"
      >
        <Info className="w-4 h-4 shrink-0" />
        {titles.length ? (
          <div className="flex items-center gap-2">
            {titles.map((t, i) => (
              <span key={i} className="text-sm">
                {t}
                {i < titles.length - 1 ? " ·" : ""}
              </span>
            ))}
            <button
              onClick={onExport}
              title="Export via email"
              className="ml-2 inline-flex items-center justify-center w-8 h-8 rounded-md border border-indigo-200 hover:bg-indigo-100"
            >
              <Mail className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <span className="text-sm">Reading list is empty</span>
        )}
      </div>
    </>
  );
}
