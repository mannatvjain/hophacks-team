// DetailsPanel.jsx
import React from "react";

/**
 * Polished card; no scroll (cap list length); titles wrap (no ellipsis).
 * References & Cited-by are NUMBERED and CLICKABLE (jump to node in graph).
 */
export default function DetailsPanel({ node, nodes, onJumpToId, className = "" }) {
  const container = "bg-white shadow-lg rounded-lg p-6 flex flex-col justify-start";
  const maxItems = 10; // keep single page

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

  return (
    <div className={`${container} ${className}`}>
      {/* Title wraps to new line, no ellipsis */}
      <h2 className="text-2xl font-semibold text-gray-900 leading-snug break-words">
        {node.title || String(node.id)}
      </h2>
      <p className="text-xs text-gray-500 break-all">{String(node.id)}</p>

      {/* Authors / Year */}
      <div className="text-sm text-gray-800 mt-3">
        <span className="font-medium">
          {Array.isArray(node.authors) ? node.authors.join(", ") : node.authors || "—"}
        </span>{" "}
        {node.year ? <span className="text-gray-600">({node.year})</span> : null}
      </div>

      {/* References (numbered & clickable) */}
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
                  {getTitle(cid)}
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

      {/* Cited by (numbered & clickable) */}
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
                  {getTitle(cid)}
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
