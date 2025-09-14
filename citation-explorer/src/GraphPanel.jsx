// src/GraphPanel.jsx
import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

/* ----------------- star geometry (outside component) ----------------- */
const STAR_OUTER = 16;  // tweak visual size of gold star
const STAR_INNER = 8;
function starPath(cx, cy, spikes = 5, outerR = STAR_OUTER, innerR = STAR_INNER) {
  let path = "";
  const step = Math.PI / spikes;
  for (let i = 0; i < spikes * 2; i++) {
    const r = i % 2 === 0 ? outerR : innerR;
    const x = cx + Math.cos(i * step - Math.PI / 2) * r;
    const y = cy + Math.sin(i * step - Math.PI / 2) * r;
    path += (i === 0 ? "M" : "L") + x + "," + y;
  }
  return path + "Z";
}

/* ----------------------------- component ----------------------------- */
export default function GraphPanel({
  data,                // { nodes, links, goldId, top10Ids, ... }
  onSelect,
  onToggleReading,
  readingIds,
  className = "",
  fitSignal = 0,       // animated fit-all
  recenterSignal = 0,  // instant recenter (keep zoom)
}) {
  const svgRef  = useRef(null);
  const zoomRef = useRef(null); // keep ONE zoom instance

  const GOLD   = "#FFD700";
  const ORANGE = "#f97316";
  const BLACK  = "#0f172a";
  const GREEN  = "#22c55e";
  const PURPLE = "#7c3aed";
  const LINK   = "#9ca3af";
  const ARROW  = "#94a3b8";
  const LABEL  = "#334155";

  useEffect(() => {
    const { nodes, links, goldId, top10Ids, shortest_distance } = data;
    
    const pathEdges = new Set(
      (shortest_distance ?? [])
        .slice(0, -1)
        .map((id, i) => `${shortest_distance[i]}->${shortest_distance[i + 1]}`)
    );    

    const svg  = d3.select(svgRef.current);
    const root = svg.select(".root");
    root.selectAll("*").remove();

    const bb = svgRef.current.getBoundingClientRect();
    const width  = Math.max(1, bb.width  || svgRef.current.clientWidth  || 800);
    const height = Math.max(1, bb.height || svgRef.current.clientHeight || 600);

    /* ---------- marker: wide wedge, scales with stroke width ---------- */
    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "arrow")
      .attr("markerUnits", "strokeWidth") // scale with line thickness
      .attr("viewBox", "0 0 12 12")
      .attr("refX", 9.5)                 // align tip at line end
      .attr("refY", 6)
      .attr("markerWidth", 5)             // size (tweak 4.5–5.5 to taste)
      .attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M 0 0 L 12 6 L 0 12 L 3 6 Z") // wide wedge to cover link edges
      .attr("fill", ARROW);

      defs.append("marker")
      .attr("id", "arrow-orange")
      .attr("markerUnits", "strokeWidth")
      .attr("viewBox", "0 0 12 12")
      .attr("refX", 9.5)
      .attr("refY", 6)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M 0 0 L 12 6 L 0 12 L 3 6 Z")
      .attr("fill", ORANGE);
    

    /* ---------------------------- layers ----------------------------- */
    const linkLayer  = root.append("g").attr("stroke", LINK).attr("stroke-opacity", 0.55);
    const nodeLayer  = root.append("g");
    const labelLayer = root.append("g");

    const linkStroke = 1.25;

    const link = linkLayer.selectAll("line")
      .data(links, d => `${d.source}->${d.target}`)
      .join("line")
      .attr("stroke-width", linkStroke)
      .attr("stroke-linecap", "butt")
      .attr("stroke", d => {
        const key = `${String(d.source.id ?? d.source)}->${String(d.target.id ?? d.target)}`;
        return pathEdges.has(key) ? ORANGE : LINK;
      })
      .attr("marker-end", d => {
        const key = `${String(d.source.id ?? d.source)}->${String(d.target.id ?? d.target)}`;
        return pathEdges.has(key) ? "url(#arrow-orange)" : "url(#arrow)";
      });

    const nodeRadius  = 5;
    const hoverRadius = 7;
    const top10Radius = 8;
    const top10Hover = 10;
    const gapOffset   = 4;                // modest clearance from node to link

    /* ------------------------- color helpers ------------------------- */
    const isPurpleNow = (d) =>
      d.__purple === true || (readingIds?.has(String(d.id)) ?? false);
    const isGold  = (d) => goldId && String(d.id) === goldId;
    const inTop10 = (d) => top10Ids.has(String(d.id));

    const scoreFill = (d) =>
      isGold(d)  ? GOLD :
      inTop10(d) ? ORANGE :
      BLACK;

    const nodeFill = (d) => (isPurpleNow(d) ? PURPLE : scoreFill(d));

    /* ---------------------- gold star (path) node --------------------- */
    let sim; // define before drag handlers

    const goldNodes = nodeLayer.selectAll("path.gold")
      .data(nodes.filter(isGold), d => String(d.id))
      .join(enter => enter.append("path").attr("class", "gold"))
      .attr("d", d => starPath(d.x || 0, d.y || 0))
      .attr("fill", (d) => (isPurpleNow(d) ? PURPLE : GOLD))
      .attr("stroke", "white")
      .attr("stroke-width", 1.6)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        if (event.metaKey || event.ctrlKey) {
          d.__purple = !isPurpleNow(d);
          d3.select(event.currentTarget).attr("fill", d.__purple ? PURPLE : GOLD);
          onToggleReading && onToggleReading(d.id);
        } else {
          onSelect && onSelect(d);
        }
      })
      .on("mouseover", function (_, d) {
        d3.select(this).attr("fill", GREEN);
        const id = String(d.id);
        link.attr("stroke-opacity", (L) =>
          String(L.source.id ?? L.source) === id || String(L.target.id ?? L.target) === id ? 0.9 : 0.15
        );
        node.attr("opacity", (n) => (n === d ? 1 : 0.35));
        label.filter((ld) => ld === d).attr("opacity", 1);
      })
      .on("mouseout", function (_, d) {
        d3.select(this).attr("fill", d.__purple ? PURPLE : GOLD);
        link.attr("stroke-opacity", 0.55);
        node.attr("opacity", 1);
        label.attr("opacity", 0);
      })
      .call(
        d3.drag()
          .on("start", (event, d) => { if (sim) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag",  (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end",   (event, d) => { d.fx = null;  d.fy = null; if (sim) sim.alphaTarget(0); })
      );

    /* ------------------------- normal circles ------------------------ */
    const node = nodeLayer.selectAll("circle")
      .data(nodes.filter(d => !isGold(d)), d => String(d.id))
      .join("circle")
      .attr("r", d => inTop10(d) ? top10Radius : nodeRadius)
      .attr("fill", nodeFill)
      .attr("stroke", "white")
      .attr("stroke-width", 1.6)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        if (event.metaKey || event.ctrlKey) {
          d.__purple = !isPurpleNow(d);
          d3.select(event.currentTarget).attr("fill", nodeFill(d));
          onToggleReading && onToggleReading(d.id);
        } else {
          onSelect && onSelect(d);
        }
      })
      .on("mouseover", function (_, d) {
        d3.select(this).transition().duration(120).attr("r", inTop10(d) ? top10Hover : hoverRadius).attr("fill", GREEN);
        const id = String(d.id);
        link.attr("stroke-opacity", (L) =>
          String(L.source.id ?? L.source) === id || String(L.target.id ?? L.target) === id ? 0.9 : 0.15
        );
        node.attr("opacity", (n) => (n === d ? 1 : 0.35));
        label.filter((ld) => ld === d).attr("opacity", 1);
      })
      .on("mouseout", function (_, d) {
        d3.select(this).transition().duration(120).attr("r", inTop10(d) ? top10Radius : nodeRadius).attr("fill", nodeFill(d));
        link.attr("stroke-opacity", 0.55);
        node.attr("opacity", 1);
        label.attr("opacity", 0);
      })
      .call(
        d3.drag()
          .on("start", (event, d) => { if (sim) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag",  (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end",   (event, d) => { d.fx = null;  d.fy = null; if (sim) sim.alphaTarget(0); })
      );

    /* ---------------------------- labels ----------------------------- */
    const labelText = (d) =>
      d.title ? (d.title.length > 42 ? d.title.slice(0, 41) + "…" : d.title) : String(d.id);

    const label = labelLayer.selectAll("text")
      .data(nodes, d => String(d.id))
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

    /* --------------------------- zoom/pan ---------------------------- */
    const zoom = d3.zoom()
      .scaleExtent([0.3, 6])
      .on("start", () => svg.style("cursor", "grabbing"))
      .on("end",   () => svg.style("cursor", "default"))
      .on("zoom",  (e) => root.attr("transform", e.transform));
    zoomRef.current = zoom;
    svg.call(zoom).style("cursor", "default");

    /* ---------------------- angular spread force --------------------- */
    function forceAngularSpread(minSep = 0.35, strength = 0.03) {
      let nodeById, adj;
      function init() {
        nodeById = new Map(nodes.map((n) => [String(n.id), n]));
        adj = new Map(nodes.map((n) => [n, []]));
        for (const l of links) {
          const s = nodeById.get(String(l.source.id ?? l.source));
          const t = nodeById.get(String(l.target.id ?? l.target));
          if (s && t) { adj.get(s).push(t); adj.get(t).push(s); }
        }
      }
      function force() {
        for (const v of nodes) {
          const neigh = adj.get(v);
          if (!neigh || neigh.length < 2) continue;
          const items = neigh
            .map((n) => {
              const dx = (n.x ?? 0) - (v.x ?? 0);
              const dy = (n.y ?? 0) - (v.y ?? 0);
              const a  = Math.atan2(dy, dx);
              return { n, a, dx, dy };
            })
            .sort((a, b) => a.a - b.a);
          for (let i = 0; i < items.length; i++) {
            const A = items[i];
            const B = items[(i + 1) % items.length];
            let delta = B.a - A.a;
            if (delta <= 0) delta += Math.PI * 2;
            if (delta < minSep) {
              const push = (minSep - delta) * strength;
              const ra = Math.hypot(A.dx, A.dy) || 1;
              const rb = Math.hypot(B.dx, B.dy) || 1;
              const tax =  (A.dy / ra), tay = -(A.dx / ra);
              const tbx = -(B.dy / rb), tby =  (B.dx / rb);
              A.n.vx = (A.n.vx || 0) + tax * push;  A.n.vy = (A.n.vy || 0) + tay * push;
              B.n.vx = (B.n.vx || 0) + tbx * push;  B.n.vy = (B.n.vy || 0) + tby * push;
              v.vx   = (v.vx   || 0) - (tax + tbx) * (push * 0.5);
              v.vy   = (v.vy   || 0) - (tay + tby) * (push * 0.5);
            }
          }
        }
      }
      force.initialize = init;
      return force;
    }

    /* ---------------------------- forces ----------------------------- */
    sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => String(d.id)).distance(48).strength(0.85))
      .force("charge", d3.forceManyBody().strength(-180).distanceMax(450))
      .force("x", d3.forceX(0).strength(0.05))
      .force("y", d3.forceY(0).strength(0.05))

      .force("center", d3.forceCenter(0, 0))

      .force("collision", d3.forceCollide().radius(d => (inTop10(d) ? top10Radius : nodeRadius) + 3).iterations(2))
      .force("angular", forceAngularSpread(0.35, 0.03))
      .velocityDecay(0.25)
      .alpha(1)
      .alphaDecay(0.06);

    /* ---- endpoint math so arrow covers link end, extra near star ---- */
    function endpointWithGap(d, which) {
      const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y;
      const dx = tx - sx, dy = ty - sy, dist = Math.hypot(dx, dy) || 1;
      const ux = dx / dist, uy = dy / dist;

      const srcBase = nodeRadius + gapOffset;
      const tarBase = nodeRadius + gapOffset + linkStroke * 0.5; // hide half-stroke under marker

      //very scrappy patch: if i subtracted \infty from the star then it'd stop the lines right outside the star (because of the white barrier. so we can tune how far by just fiddling with the negatives.)
      const srcExtra = (goldId && String(d.source.id ?? d.source) === goldId) ? (STAR_OUTER - 6) : 0;
      const tarExtra = (goldId && String(d.target.id ?? d.target) === goldId) ? (STAR_OUTER - 6) : 0;

      const srcR = srcBase + srcExtra;
      const tarR = tarBase + tarExtra;

      if (which === "x1") return sx + ux * srcR;
      if (which === "y1") return sy + uy * srcR;
      if (which === "x2") return tx - ux * tarR;
      if (which === "y2") return ty - uy * tarR;
    }

    /* ----------------------------- tick ------------------------------ */
    sim.on("tick", () => {
      link
        .attr("x1", d => endpointWithGap(d, "x1"))
        .attr("y1", d => endpointWithGap(d, "y1"))
        .attr("x2", d => endpointWithGap(d, "x2"))
        .attr("y2", d => endpointWithGap(d, "y2"));

      goldNodes.attr("d", d => starPath(d.x, d.y));
      node.attr("cx", d => d.x).attr("cy", d => d.y);
      label.attr("x", d => (d.x ?? 0) + 9).attr("y", d => (d.y ?? 0) + 4);
    });

    /* ------------------------- initial center ------------------------ */
    immediateFit(svg, zoomRef.current, nodes, width, height, 32);

    /* --------------------- external camera signals ------------------- */
    svg.on("smartfit", () => animateFit(svg, zoomRef.current, nodes, width, height, 32));
    svg.on("recenter", () => recenterOnly(svg, zoomRef.current, nodes, width, height));

    return () => {
      svg.on("smartfit", null).on("recenter", null);
      sim.stop();
    };
  // do NOT include readingIds here (prevents sim rebuilds/jitter)
  }, [data]);

  /* ---------------- recolor circles + star on reading list change ---------------- */
  useEffect(() => {
    const svg = d3.select(svgRef.current);

    // circles
    svg.select(".root").selectAll("circle").each(function (d) {
      d.__purple = readingIds?.has(String(d.id));
      // gold/top10/black are handled by nodeFill; here purple wins if set
      const isPurple = d.__purple === true;
      if (isPurple) d3.select(this).attr("fill", "#7c3aed");
    });

    // gold star
    svg.select(".root").selectAll("path.gold").each(function (d) {
      d.__purple = readingIds?.has(String(d.id));
      d3.select(this).attr("fill", d.__purple ? "#7c3aed" : "#FFD700");
    });
  }, [readingIds, data.top10Ids, data.goldId]);

  /* ------------------------- camera triggers ------------------------- */
  useEffect(() => { d3.select(svgRef.current).dispatch("smartfit"); }, [fitSignal]);
  useEffect(() => { d3.select(svgRef.current).dispatch("recenter"); }, [recenterSignal]);

  return (
    <svg ref={svgRef} className={`w-full h-full rounded-lg bg-white ${className}`}>
      <g className="root" />
    </svg>
  );
}

/* ============================ fit helpers ============================ */
function computeBounds(nodes) {
  const xs = nodes.map(n => n.x || 0), ys = nodes.map(n => n.y || 0);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  return { minX, maxX, minY, maxY, w: Math.max(1, maxX - minX), h: Math.max(1, maxY - minY) };
}

function immediateFit(svg, zoom, nodes, width, height, pad = 32) {
  const { minX, maxX, minY, maxY, w, h } = computeBounds(nodes);
  const k = Math.min(1, 0.9 / Math.max(w / (width - pad * 2), h / (height - pad * 2)));
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2 - k * cx, height / 2 - k * cy).scale(k));
}

function animateFit(svg, zoom, nodes, width, height, pad = 32) {
  const t0 = d3.zoomTransform(svg.node());
  const { minX, maxX, minY, maxY, w, h } = computeBounds(nodes);
  const k = Math.min(1, 0.9 / Math.max(w / (width - pad * 2), h / (height - pad * 2)));
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const t  = d3.zoomIdentity.translate(width / 2 - k * cx, height / 2 - k * cy).scale(k);
  const dist = Math.hypot(t0.x - t.x, t0.y - t.y) + Math.abs(t0.k - t.k) * 50;
  if (dist < 20) return;
  svg.transition().duration(500).call(zoom.transform, t);
}

function recenterOnly(svg, zoom, nodes, width, height) {
  const t0 = d3.zoomTransform(svg.node());
  const { minX, maxX, minY, maxY } = computeBounds(nodes);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2 - t0.k * cx, height / 2 - t0.k * cy).scale(t0.k));
}
