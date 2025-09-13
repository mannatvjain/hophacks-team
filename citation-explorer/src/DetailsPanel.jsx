// DetailsPanel.jsx
import React from "react";

/**
 * Polished card design (as before), but stays single-page:
 * - We cap list lengths and truncate long strings to avoid scrolling.
 */
export default function DetailsPanel({ node, nodes, className = "" }) {
  const container = "bg-white shadow-lg rounded-lg p-6 flex flex-col justify-start";
  const maxItems = 8; // cap lists to avoid scroll
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

  return (
    <div className={`${container} ${className}`}>
      {/* Title + ID */}
      <h2 className="text-2xl font-semibold text-gray-900 truncate" title={node.title}>
        {trunc(node.title || String(node.id), 64)}
      </h2>
      <p className="text-xs text-gray-500 truncate" title={String(node.id)}>
        {trunc(String(node.id), 72)}
      </p>

      {/* Authors / Year / Abstract (compact) */}
      <div className="text-sm text-gray-800 mt-3">
        <span className="font-medium">
          {Array.isArray(node.authors)
            ? trunc(node.authors.join(", "), 90)
            : trunc(node.authors, 90)}
        </span>{" "}
        {node.year ? <span className="text-gray-600">({node.year})</span> : null}
      </div>
      {node.abstract ? (
        <p className="text-sm text-gray-800 mt-2">{trunc(node.abstract, 180)}</p>
      ) : null}

      {/* References */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <h3 className="text-gray-800 font-medium mb-2">
          References {refs.length > maxItems ? `(${refs.length})` : ""}
        </h3>
        {refs.length === 0 ? (
          <p className="text-sm text-gray-600">None</p>
        ) : (
          <ul className="list-disc list-inside text-sm text-gray-800 space-y-1">
            {refs.slice(0, maxItems).map((cid) => (
              <li key={cid} className="truncate" title={getTitle(cid)}>
                {trunc(getTitle(cid), 70)}
              </li>
            ))}
            {refs.length > maxItems ? (
              <li className="text-xs text-gray-500">
                …and {refs.length - maxItems} more
              </li>
            ) : null}
          </ul>
        )}
      </div>

      {/* Cited by (now computed; should not show "None" unless truly none) */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <h3 className="text-gray-800 font-medium mb-2">
          Cited by {cites.length > maxItems ? `(${cites.length})` : ""}
        </h3>
        {cites.length === 0 ? (
          <p className="text-sm text-gray-600">None</p>
        ) : (
          <ul className="list-disc list-inside text-sm text-gray-800 space-y-1">
            {cites.slice(0, maxItems).map((cid) => (
              <li key={cid} className="truncate" title={getTitle(cid)}>
                {trunc(getTitle(cid), 70)}
              </li>
            ))}
            {cites.length > maxItems ? (
              <li className="text-xs text-gray-500">
                …and {cites.length - maxItems} more
              </li>
            ) : null}
          </ul>
        )}
      </div>
    </div>
  );
}
