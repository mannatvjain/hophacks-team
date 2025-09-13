// GraphPanel.jsx
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

/**
 * Centered force graph:
 * - The centroid of all nodes is always aligned to the center of the viewport.
 * - Zoom changes scale only; panning is disabled (we auto-center).
 * Observable-esque polish:
 * - Light, semi-transparent links; crisp arrowheads.
 * - Neighbor fade on hover; labels show on hover only with white halo.
 * Colors:
 * - Orange for highest-degree nodes, black for others.
 * - Rowan green for hover/active highlight.
 */
export default function GraphPanel({ data, onSelect, className = "" }) {
  const svgRef = useRef(null);

  // palette
  const ORANGE = "#f97316"; // orange-500 (max-degree nodes)
  const BLACK  = "#0f172a"; // slate-900 (others)
  const GREEN  = "#22c55e"; // green-500 (Rowan-like accent on hover/active)
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

    // --- defs: arrowheads
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

    // --- layers
    const linkLayer  = root.append("g").attr("stroke", LINK).attr("stroke-opacity", 0.55);
    const nodeLayer  = root.append("g");
    const labelLayer = root.append("g");

    // --- links (straight) with arrowheads
    const link = linkLayer.selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", 1.25)
      .attr("marker-end", "url(#arrow)");

    // --- node color: orange if max-degree else black
    const nodeFill = (d) => (degree.get(String(d.id)) === maxDeg ? ORANGE : BLACK);

    // --- nodes
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
        // neighbor fade
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
            d.fx = null; d.fy = null; // release for lively physics
          })
      );

    // --- labels (hover-only) with white halo
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

    // --- zoom: scale only; we keep centroid centered (no panning)
    let k = 1;
    const zoom = d3.zoom()
      .scaleExtent([0.3, 6])
      .on("zoom", (event) => {
        k = event.transform.k;
        recenter(); // re-center at new scale
      });
    svg.call(zoom).on("dblclick.zoom", null); // disable dblclick to keep calm

    // --- centering: keep centroid at viewport center
    function recenter() {
      if (!nodes.length) return;
      const cx = d3.mean(nodes, (n) => n.x || 0) || 0;
      const cy = d3.mean(nodes, (n) => n.y || 0) || 0;
      const tx = width / 2  - k * cx;
      const ty = height / 2 - k * cy;
      root.attr("transform", `translate(${tx},${ty}) scale(${k})`);
    }

    // --- forces (Observable-ish)
    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => String(d.id)).distance(70).strength(0.55))
      .force("charge", d3.forceManyBody().strength(-320))
      .force("center", d3.forceCenter(0, 0)) // center in simulation space (we recenter visually)
      .force("collision", d3.forceCollide().radius(12).iterations(2))
      .velocityDecay(0.25)
      .alpha(1)
      .alphaDecay(0.06);

    sim.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);

      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

      // place labels slightly offset from node
      label.attr("x", (d) => d.x + 10).attr("y", (d) => d.y + 4);

      // keep graph centered each tick
      recenter();
    });

    // initial gentle fit via scale (optional). comment out if you prefer no auto-zoom.
    // d3.timeout(() => svg.transition().duration(600).call(zoom.scaleTo, 1), 400);

    return () => { sim.stop(); };
  }, [data, onSelect]);

  return (
    <svg ref={svgRef} className={`w-full h-full rounded-lg bg-white ${className}`}>
      <g className="root" />
    </svg>
  );
}
