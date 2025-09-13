// GraphPanel.jsx
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

/**
 * Centered force graph:
 * - Centroid pinned to center; when selectedId changes we center that node.
 * Observable-esque polish:
 * - Light links, crisp arrows, neighbor fade, labels on hover (white halo).
 * Colors:
 * - Orange for highest-degree nodes, black for others.
 * - Rowan green on hover/active.
 */
export default function GraphPanel({ data, selectedId, onSelect, className = "" }) {
  const svgRef = useRef(null);

  const ORANGE = "#f97316"; // orange-500 (max-degree nodes)
  const BLACK  = "#0f172a"; // slate-900
  const GREEN  = "#22c55e"; // green-500 (Rowan-like)
  const LINK   = "#9ca3af"; // gray-400
  const ARROW  = "#94a3b8"; // slate-400
  const LABEL  = "#334155"; // slate-700

  useEffect(() => {
    const { nodes, links, degree, maxDeg } = data;

    const svg  = d3.select(svgRef.current);
    const root = svg.select(".root");
    root.selectAll("*").remove();

    const width  = svgRef.current.clientWidth  || 800;
    const height = svgRef.current.clientHeight || 600;

    // defs
    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 16)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", ARROW)
      .attr("opacity", 0.9);

    // layers
    const linkLayer  = root.append("g").attr("stroke", LINK).attr("stroke-opacity", 0.55);
    const nodeLayer  = root.append("g");
    const labelLayer = root.append("g");

    // links
    const link = linkLayer.selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", 1.25)
      .attr("marker-end", "url(#arrow)");

    // coloring
    const nodeFill = (d) => (degree.get(String(d.id)) === maxDeg ? ORANGE : BLACK);

    // nodes
    const node = nodeLayer.selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", 5)
      .attr("fill", nodeFill)
      .attr("stroke", "white")
      .attr("stroke-width", 1.2)
      .style("cursor", "pointer")
      .on("click", (_, d) => onSelect && onSelect(d))
      .on("mouseover", function (_, d) {
        d3.select(this).transition().duration(120).attr("r", 7).attr("fill", GREEN);
        const id = String(d.id);
        link.attr("stroke-opacity", (L) =>
          String(L.source.id ?? L.source) === id || String(L.target.id ?? L.target) === id ? 0.9 : 0.15
        );
        node.attr("opacity", (n) => (n === d ? 1 : 0.35));
        label.filter((ld) => ld === d).attr("opacity", 1);
      })
      .on("mouseout", function () {
        d3.select(this).transition().duration(120).attr("r", 5).attr("fill", (d) => nodeFill(d));
        link.attr("stroke-opacity", 0.55);
        node.attr("opacity", 1);
        label.attr("opacity", 0);
      })
      .call(
        d3.drag()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x; d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      );

    // labels (hover only)
    const labelText = (d) =>
      d.title ? (d.title.length > 42 ? d.title.slice(0, 41) + "…" : d.title) : String(d.id);
    const label = labelLayer.selectAll("text")
      .data(nodes)
      .join("text")
      .text(labelText)
      .attr("font-size", 11)
      .attr("fill", LABEL)
      .attr("opacity", 0)
      .style("paint-order", "stroke")
      .style("stroke", "white")
      .style("stroke-width", 3)
      .style("stroke-linejoin", "round")
      .attr("pointer-events", "none");

    // zoom = scale only (no panning); we keep center ourselves
    let k = 1;
    const zoom = d3.zoom()
      .scaleExtent([0.3, 6])
      .on("zoom", (event) => { k = event.transform.k; recenter(); });
    svg.call(zoom).on("dblclick.zoom", null);

    // keep centroid in center; if selectedId provided, center that node instead
    function recenter() {
      if (!nodes.length) return;
      let cx, cy;
      if (selectedId) {
        const target = nodes.find((n) => String(n.id) === String(selectedId));
        cx = target?.x ?? 0;
        cy = target?.y ?? 0;
      } else {
        cx = d3.mean(nodes, (n) => n.x || 0) || 0;
        cy = d3.mean(nodes, (n) => n.y || 0) || 0;
      }
      const tx = width / 2  - k * cx;
      const ty = height / 2 - k * cy;
      root.attr("transform", `translate(${tx},${ty}) scale(${k})`);
    }

    // forces
    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => String(d.id)).distance(70).strength(0.55))
      .force("charge", d3.forceManyBody().strength(-320))
      .force("center", d3.forceCenter(0, 0)) // visual centering handled in recenter()
      .force("collision", d3.forceCollide().radius(12).iterations(2))
      .velocityDecay(0.25)
      .alpha(1)
      .alphaDecay(0.06);

    sim.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
      label.attr("x", (d) => d.x + 10).attr("y", (d) => d.y + 4);
      recenter();
    });

    // if we mount with a preselected id, center right away after a short settle
    if (selectedId) setTimeout(recenter, 300);

    return () => { sim.stop(); };
  }, [data, selectedId, onSelect]);

  return (
    <svg ref={svgRef} className={`w-full h-full rounded-lg bg-white ${className}`}>
      <g className="root" />
    </svg>
  );
}
