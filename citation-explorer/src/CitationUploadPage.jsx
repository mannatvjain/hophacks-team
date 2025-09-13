import React, { useState } from "react";
import { Upload, Link as LinkIcon } from "lucide-react";

export default function CitationUploadPage() {
  const [doi, setDoi] = useState("");
  const [fileName, setFileName] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) setFileName(file.name);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
  
    const payload = {
      doi: doi.trim() || null,
      file: fileName || null,
    };
  
    // For now, just show it in the browser console.
    console.log("Submitting:", payload);
  
    // TODO: send `payload` to your backend API with fetch()
  };
  

  return (
    <div className="w-full h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-white p-6">
      <div className="max-w-xl w-full bg-white shadow-xl rounded-3xl p-10 flex flex-col items-center gap-8">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
          Start Your Citation Journey
        </h1>

        <form onSubmit={handleSubmit} className="w-full flex flex-col gap-6">
          {/* DOI Input */}
          <div className="flex items-center gap-3 border-b border-slate-300 focus-within:border-slate-500">
            <LinkIcon className="w-5 h-5 text-slate-400" />
            <input
              type="text"
              value={doi}
              onChange={(e) => setDoi(e.target.value)}
              placeholder="Enter DOI (e.g., 10.1038/s41586-020-03167-3)"
              className="flex-1 py-2 px-1 bg-transparent focus:outline-none text-slate-700"
            />
          </div>

          {/* File Upload */}
          <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 hover:border-slate-400 rounded-2xl py-10 cursor-pointer transition">
            <Upload className="w-6 h-6 text-slate-400 mb-2" />
            <span className="text-sm text-slate-500">
              {fileName ? fileName : "Upload PDF of paper"}
            </span>
            <input
              type="file"
              accept="application/pdf"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          <button
            type="submit"
            className="mt-4 w-full py-3 rounded-2xl bg-slate-900 text-white font-medium tracking-wide hover:bg-slate-800 transition"
          >
            Continue
          </button>
        </form>
      </div>
    </div>
  );
  
}

export function CitationRightPane({ data = demoData }) {
    const [selected, setSelected] = React.useState(null);
  
    // compute simple in-degree (how many times each paper is cited in this dataset)
    const inDegree = React.useMemo(() => {
      const counts = new Map(data.nodes.map(n => [String(n.id), 0]));
      for (const n of data.nodes) {
        for (const t of n.outCitations || []) {
          const k = String(t);
          if (counts.has(k)) counts.set(k, counts.get(k) + 1);
        }
      }
      return counts;
    }, [data]);
  
    return (
      <div className="w-full h-screen grid grid-cols-[1fr_320px] gap-4 p-6 bg-white">
        {/* Graph container (placeholder for now) */}
        <div className="rounded-3xl border border-slate-200 bg-white shadow-sm p-4 flex flex-col">
          <div className="text-sm text-slate-500 mb-3">
            graph visualization (placeholder)
          </div>
          <div className="flex-1 rounded-2xl border border-dashed border-slate-300 p-4 overflow-auto">
            <div className="text-xs text-slate-500 mb-2">Click to preview details:</div>
            <ul className="space-y-2">
              {data.nodes.map(n => (
                <li key={n.id}>
                  <button
                    className={`w-full text-left px-3 py-2 rounded-xl border ${
                      selected?.id === n.id
                        ? "bg-slate-900 text-white border-slate-900"
                        : "bg-white hover:bg-slate-50 border-slate-200"
                    }`}
                    onClick={() => setSelected(n)}
                  >
                    <div className="text-sm font-medium truncate">
                      {n.title || n.id}
                    </div>
                    <div className="text-xs opacity-70 truncate">{String(n.id)}</div>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
  
        {/* Details sidebar */}
        <NodeDetails node={selected} inDegree={inDegree} />
      </div>
    );
  }
  
  function NodeDetails({ node, inDegree }) {
    if (!node) {
      return (
        <aside className="rounded-3xl border border-slate-200 bg-white shadow-sm p-5">
          <h2 className="text-lg font-semibold tracking-tight text-slate-900 mb-4">
            Paper details
          </h2>
          <p className="text-sm text-slate-500">
            Select a paper in the graph to see its metadata and links.
          </p>
        </aside>
      );
    }
  
    const cites = node.outCitations || [];
    const citedByCount = inDegree.get(String(node.id)) || 0;
  
    return (
      <aside className="rounded-3xl border border-slate-200 bg-white shadow-sm p-5 flex flex-col gap-4 min-h-0">
        <h2 className="text-lg font-semibold tracking-tight text-slate-900">
          Paper details
        </h2>
        <div className="space-y-1">
          <div className="text-sm font-medium text-slate-900 break-words">
            {node.title || "Untitled"}
          </div>
          <div className="text-xs text-slate-500 break-all">{String(node.id)}</div>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center">
          <Stat label="Year" value={node.year ?? "—"} />
          <Stat label="Cites" value={cites.length} />
          <Stat label="Cited by*" value={citedByCount} note="*in this dataset" />
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-1">
            Authors
          </div>
          <div className="text-sm text-slate-800 whitespace-pre-wrap">
            {Array.isArray(node.authors)
              ? node.authors.join(", ")
              : node.authors || "—"}
          </div>
        </div>
        <div className="min-h-0 overflow-auto">
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-1">
            References
          </div>
          {cites.length === 0 ? (
            <div className="text-sm text-slate-500">None</div>
          ) : (
            <ul className="space-y-2">
              {cites.map(cid => (
                <li key={cid} className="text-sm break-all text-slate-700">
                  {String(cid)}
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    );
  }
  
  function Stat({ label, value, note }) {
    return (
      <div className="rounded-2xl border border-slate-200 p-3">
        <div className="text-base font-semibold text-slate-900">{value}</div>
        <div className="text-[11px] text-slate-500">{label}</div>
        {note ? (
          <div className="text-[10px] text-slate-400 mt-1">{note}</div>
        ) : null}
      </div>
    );
  }
  
  // demo dataset
  const demoData = {
    nodes: [
      {
        id: "10.1038/s41586-020-03167-3",
        title: "Original Research Paper",
        year: 2021,
        authors: ["Smith J", "Lee K"],
        outCitations: [
          "10.1126/science.aaz1776",
          "10.1016/j.cell.2020.12.015",
        ],
      },
      {
        id: "10.1126/science.aaz1776",
        title: "Background Study A",
        year: 2019,
        authors: ["Patel R"],
        outCitations: [],
      },
      {
        id: "10.1016/j.cell.2020.12.015",
        title: "Background Study B",
        year: 2020,
        authors: ["Chen X", "Ng M"],
        outCitations: [],
      },
    ],
  };
  