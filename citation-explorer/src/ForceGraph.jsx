import React, { useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";

/**
 * Minimal force-directed graph.
 * Props:
 *  - data: { nodes: [{ id, title?, outCitations?: string[] }], ...optional }
 *  - width, height: numbers (defaults provided)
 *  - onSelect: function(node) -> called when a node is clicked
 */
export default function ForceGraph({
  data,
  width = 900,
  height = 600,
  onSelect = () => {},
}) {
  const svgRef = useRef(null);
  const gRef = useRef(null);

  // Normalize: copy nodes; build links from outCitations that point to existing nodes
  const { nodes, links } = useMemo(() => {
    const nodes = (data?.nodes ?? []).map((d) => ({ ...d }));
    const idSet = new Set(nodes.map((d) => String(d.id)));
    const links = [];

    for (const src of nodes) {
      const outs = Array.isArray(src.outCitations) ? src.outCitations : [];
      for (const tid of outs) {
        const t = String(tid);
        if (idSet.has(t)) links.push({ source: String(src.id), target: t });
      }
    }
    return { nodes, links };
  }, [data]);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const g = d3.select(gRef.current);

    // clean previous contents on re-render
    g.selectAll("*").remove();

    svg.attr("viewBox", [0, 0, width, height]);

    // simulation
    const sim = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3.forceLink(links).id((d) => String(d.id)).distance(80)
      )
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2));

    // links
    const link = g
      .append("g")
      .attr("stroke", "#d1d5db") // gray-300
      .attr("stroke-width", 1.5)
      .selectAll("line")
      .data(links)
      .join("line");

    // nodes
    const node = g
      .append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", 6)
      .attr("fill", "#3b82f6") // blue-500
      .style("cursor", "pointer")
      .on("click", (_, d) => onSelect(d))
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
          })
      );

    // labels (tiny)
    const label = g
      .append("g")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .text((d) => (d.title ? truncate(d.title, 32) : String(d.id)))
      .attr("font-size", 10)
      .attr("fill", "#374151"); // gray-700

    sim.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

      label
        .attr("x", (d) => d.x + 8)
        .attr("y", (d) => d.y + 4);
    });

    return () => sim.stop();
  }, [nodes, links, width, height, onSelect]);

  return (
    <svg
      ref={svgRef}
      className="w-full h-full rounded-2xl bg-white shadow"
      width={width}
      height={height}
    >
      <g ref={gRef} />
    </svg>
  );
}

function truncate(s, n) {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}
