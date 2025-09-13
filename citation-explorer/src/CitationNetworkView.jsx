// CitationNetworkView.jsx
import React, { useMemo, useState } from "react";
import { ArrowLeft, CheckCircle2, RotateCcw, Search, Info, Mail } from "lucide-react";
import GraphPanel from "./GraphPanel";
import DetailsPanel from "./DetailsPanel";

export default function CitationNetworkView({ initialData, onBack }) {
  const [selectedNode, setSelectedNode] = useState(null);
  const [fitSignal, setFitSignal] = useState(0);         // animated fit (Focus)
  const [recenterSignal, setRecenterSignal] = useState(0); // instant recenter (Refresh)
  const [readingIds, setReadingIds] = useState(new Set()); // persistent purple list

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
    const subject = encodeURIComponent("Reading list");
    const body = encodeURIComponent(readingTitles.map((t, i) => `${i + 1}. ${t}`).join("\n"));
    window.location.href = `mailto:?subject=${subject}&body=${body}`;
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

      <div className="h-[calc(100vh-11rem)] grid grid-cols-[1fr_360px] gap-6">
        <div className="bg-white rounded-lg shadow-sm h-full">
          <GraphPanel
            data={data}
            onSelect={setSelectedNode}
            onToggleReading={toggleReading}
            readingIds={readingIds}
            fitSignal={fitSignal}
            recenterSignal={recenterSignal}
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

      {/* Reading strip — back to the original info-bar palette, thin height, scrollbar hidden */}
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
