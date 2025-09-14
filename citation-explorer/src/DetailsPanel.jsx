// DetailsPanel.jsx
import React, { useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";

/**
 * Polished card; single-page (no scroll).
 * Adds "Abstract" section with a Notion-style callout and a collapse/expand toggle.
 * - Title wraps (no ellipsis)
 * - DOI/id shown under title
 * - Authors + year
 * - Abstract (preview by default; toggle to reveal rest)
 * - References / Cited by: numbered & clickable via onJumpToId
 */

export default function DetailsPanel({ node, nodes, onJumpToId, className = "" }) {
  const container = "bg-white shadow-lg rounded-lg p-6 flex flex-col justify-start";
  const maxItems = 10; // cap list lengths to avoid scrolling
  const trunc = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || "");

  const getTitle = (id) => {
    const ref = nodes.find((n) => String(n.id) === String(id));
    return ref?.title || String(id);
  };

  if (!node) {
    return (
      <div className={`${container} ${className}`}>
        <h2 className="text-xl font-semibold text-gray-900">Paper Details</h2>
        <p className="text-sm text-gray-600 mt-2">
          Select a paper node in the graph to see its details here.
        </p>
      </div>
    );
  }

  const refs  = Array.isArray(node.outCitations) ? node.outCitations : [];
  const cites = Array.isArray(node.inCitations)  ? node.inCitations  : [];

  // --- Abstract preview / collapse logic ---
  const RAW_ABS = node?.abstract || "";
  const PREVIEW_CHARS = 420; // how much to show initially (tweak to fit your pane)

  const { preview, rest, isLong } = useMemo(() => {
    if (!RAW_ABS) return { preview: "", rest: "", isLong: false };
    if (RAW_ABS.length <= PREVIEW_CHARS) return { preview: RAW_ABS, rest: "", isLong: false };

    const hard = RAW_ABS.slice(0, PREVIEW_CHARS);
    // find a nicer cut near the end (space or period) so we don't chop mid-word
    const tail = hard.slice(-60);
    const cutAt = Math.max(tail.lastIndexOf(". "), tail.lastIndexOf(" "), 0);
    const niceCut = PREVIEW_CHARS - (tail.length - cutAt);
    const p = RAW_ABS.slice(0, Math.max(120, niceCut)); // never cut too early
    return { preview: p, rest: RAW_ABS.slice(p.length), isLong: true };
  }, [RAW_ABS]);

  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`${container} ${className}`}>
      {/* Title + ID */}
      <h2 className="text-2xl font-semibold text-gray-900 leading-snug break-words">
        {node.title || String(node.id)}
      </h2>
      <p className="text-xs text-gray-500 break-all">{String(node.id)}</p>

      {/* Authors / Year */}
      <div className="text-sm text-gray-800 mt-3">
        <span className="font-medium">
          {Array.isArray(node.authors)
            ? trunc(node.authors.join(", "), 90)
            : trunc(node.authors, 90) || "—"}
        </span>{" "}
        {node.year ? <span className="text-gray-600">({node.year})</span> : null}
      </div>

      {/* Abstract (callout with disclosure; expands a bit, then scrolls inside) */}
      <div className="mt-3">
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-gray-800 font-medium">Abstract</h3>
          <button
            onClick={() => setExpanded((v) => !v)}
            className="inline-flex items-center justify-center w-8 h-8 rounded-md border border-gray-200 bg-white hover:bg-gray-50"
            title={expanded ? "Collapse abstract" : "Expand abstract"}
            aria-expanded={expanded}
          >
            <ChevronDown
              className={`w-4 h-4 transition-transform ${expanded ? "rotate-180" : ""}`}
            />
          </button>
        </div>

        {/* Notion-like callout container */}
        <div className="relative rounded-md border border-gray-200 bg-gray-50">
          {/* Scoped slim scrollbar */}
          <style>{`
            .abstract-scroll::-webkit-scrollbar {
              width: 6px; /* total gutter space */
            }

            .abstract-scroll::-webkit-scrollbar-track {
              background: transparent;
            }

            .abstract-scroll::-webkit-scrollbar-thumb:hover {
              background: transparent;
            }

            .abstract-scroll{
              padding-right: 6px;
            }
          `}</style>

        <div
          className={`abstract-scroll p-3 text-sm text-gray-800 whitespace-pre-wrap break-words ${
            expanded ? "overflow-y-auto" : "overflow-hidden"
          }`}
          style={{
            maxHeight: expanded ? "18rem" : "7rem",
            transition: "max-height 200ms ease",
          }}
        >
          {!isLong ? (
            RAW_ABS || "—"
          ) : (
            expanded ? <>{preview}{rest}</> : <>{preview}…</>
          )}
        </div>
        </div>
      </div>

      {/* References */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <h3 className="text-gray-800 font-medium mb-2">
          References {refs.length > maxItems ? `(${refs.length})` : ""}
        </h3>
        {refs.length === 0 ? (
          <p className="text-sm text-gray-600">None</p>
        ) : (
          <ol className="list-decimal list-inside text-sm text-gray-800 space-y-1">
            {refs.slice(0, maxItems).map((cid) => (
              <li key={cid}>
                <button
                  onClick={() => onJumpToId && onJumpToId(cid)}
                  className="text-left hover:underline"
                  title="Show in graph"
                >
                  {trunc(getTitle(cid), 70)}
                </button>
              </li>
            ))}
            {refs.length > maxItems ? (
              <li className="text-xs text-gray-500">
                …and {refs.length - maxItems} more
              </li>
            ) : null}
          </ol>
        )}
      </div>

      {/* Cited by */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <h3 className="text-gray-800 font-medium mb-2">
          Cited by {cites.length > maxItems ? `(${cites.length})` : ""}
        </h3>
        {cites.length === 0 ? (
          <p className="text-sm text-gray-600">None</p>
        ) : (
          <ol className="list-decimal list-inside text-sm text-gray-800 space-y-1">
            {cites.slice(0, maxItems).map((cid) => (
              <li key={cid}>
                <button
                  onClick={() => onJumpToId && onJumpToId(cid)}
                  className="text-left hover:underline"
                  title="Show in graph"
                >
                  {trunc(getTitle(cid), 70)}
                </button>
              </li>
            ))}
            {cites.length > maxItems ? (
              <li className="text-xs text-gray-500">
                …and {cites.length - maxItems} more
              </li>
            ) : null}
          </ol>
        )}
      </div>
    </div>
  );
}
