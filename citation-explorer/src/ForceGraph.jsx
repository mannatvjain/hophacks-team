//ForceGraph.jsx
import React, { useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";

export default function ForceGraph({
    data,
    onSelect = () => {},
    width = 900,
    height = 600,
    linkDistance = 60,
    linkStrength = 0.45,
    charge = -320,
    collide = 10,
    velocityDecay = 0.25,
    alphaDecay = 0.05,
  }) {
  const svgRef = useRef(null);
  const gRef = useRef(null);

  const { nodes, links } = useMemo(() => {
    const nodes = (data?.nodes ?? []).map((d) => ({ ...d }));
    const idSet = new Set(nodes.map((d) => String(d.id)));
    const links = [];
    for (const s of nodes) {
      for (const t of s.outCitations || []) {
        const tid = String(t);
        if (idSet.has(tid)) links.push({ source: String(s.id), target: tid });
      }
    }
    return { nodes, links };
  }, [data]);

  const neighbors = useMemo(() => {
    const m = new Map(nodes.map((n) => [String(n.id), new Set([String(n.id)])]));
    for (const l of links) {
      m.get(String(l.source))?.add(String(l.target));
      m.get(String(l.target))?.add(String(l.source));
    }
    return m;
  }, [nodes, links]);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const g = d3.select(gRef.current);
    g.selectAll("*").remove();

    svg.attr("viewBox", [0, 0, width, height]).style("cursor", "grab");

    // Arrowhead
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
      .attr("fill", "#94a3b8");

    // --- Physics: brisk + playful
    const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => String(d.id)).distance(linkDistance).strength(linkStrength))
    .force("charge", d3.forceManyBody().strength(charge))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(collide).iterations(2))
    .force("x", d3.forceX(width / 2).strength(0.02))
    .force("y", d3.forceY(height / 2).strength(0.02))
    .velocityDecay(velocityDecay)
    .alpha(1)
    .alphaDecay(alphaDecay);   

    // Layers
    const link = g.append("g")
      .attr("stroke", "#cbd5e1")
      .attr("stroke-width", 1.2)
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("marker-end", "url(#arrow)");

    const node = g.append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", 6)
      .attr("fill", "#3b82f6")
      .style("cursor", "pointer")
      .call(d3.drag()
        .on("start", (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag",  (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end",   (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
      .on("click", (_, d) => onSelect(d))
      .on("mouseover", (_, d) => highlight(String(d.id), true))
      .on("mouseout",  (_, d) => highlight(String(d.id), false));

    const label = g.append("g")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .text(d => d.title ? truncate(d.title, 40) : String(d.id))
      .attr("font-size", 10)
      .attr("fill", "#475569");

    // Zoom/pan + zoom-to-fit
    const zoom = d3.zoom()
      .scaleExtent([0.25, 6])
      .on("start", () => svg.style("cursor", "grabbing"))
      .on("end",   () => svg.style("cursor", "grab"))
      .on("zoom",  (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    function zoomToFit(pad = 24) {
      if (!nodes.length) return;
      const xs = nodes.map(n => n.x ?? 0), ys = nodes.map(n => n.y ?? 0);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);
      const scale = 0.9 / Math.max(w / (width - pad * 2), h / (height - pad * 2));
      const tx = (width  - scale * (minX + maxX)) / 2;
      const ty = (height - scale * (minY + maxY)) / 2;
      svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
    }

    sim.on("tick", () => {
      link
        .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      node.attr("cx", d => d.x).attr("cy", d => d.y);
      label.attr("x", d => d.x + 8).attr("y", d => d.y + 4);
    });

    const fitTimer = setTimeout(zoomToFit, 450);

    function highlight(id, on) {
      const neigh = neighbors.get(id) || new Set([id]);
      const isN = (n) => neigh.has(String(n.id));
      node.attr("opacity", d => on ? (isN(d) ? 1 : 0.18) : 1);
      label.attr("opacity", d => on ? (isN(d) ? 1 : 0.18) : 1);
      link
        .attr("stroke", d =>
          on && (String(d.source.id ?? d.source) === id || String(d.target.id ?? d.target) === id)
            ? "#64748b" : "#cbd5e1")
        .attr("stroke-width", d =>
          on && (String(d.source.id ?? d.source) === id || String(d.target.id ?? d.target) === id)
            ? 2 : 1.2);
    }

    return () => { clearTimeout(fitTimer); sim.stop(); };
}, [
    nodes, links, neighbors, width, height, onSelect,
    linkDistance, linkStrength, charge, collide, velocityDecay, alphaDecay
  ]);  

  return (
    <svg ref={svgRef} className="w-full h-full rounded-2xl bg-white shadow" width={width} height={height}>
      <g ref={gRef} />
    </svg>
  );
}

function truncate(s, n) { return s && s.length <= n ? s : (s || "").slice(0, n - 1) + "…"; }
