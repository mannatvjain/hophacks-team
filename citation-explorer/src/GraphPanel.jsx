// GraphPanel.jsx
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function GraphPanel({ data, onSelect, className = "", fitSignal = 0 }) {
  const svgRef  = useRef(null);
  const zoomRef = useRef(null); // keep the SAME zoom instance for transforms

  const ORANGE = "#f97316";
  const BLACK  = "#0f172a";
  const GREEN  = "#22c55e";
  const LINK   = "#9ca3af";
  const ARROW  = "#94a3b8";
  const LABEL  = "#334155";

  useEffect(() => {
    const { nodes, links, degree, maxDeg } = data;

    const svg  = d3.select(svgRef.current);
    const root = svg.select(".root");
    root.selectAll("*").remove();

    // robust size read
    const bb = svgRef.current.getBoundingClientRect();
    const width  = Math.max(1, bb.width  || svgRef.current.clientWidth  || 800);
    const height = Math.max(1, bb.height || svgRef.current.clientHeight || 600);

    // --- marker (reverted style)
    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 12)            // earlier, simpler arrow
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", ARROW);

    // layers
    const linkLayer  = root.append("g").attr("stroke", LINK).attr("stroke-opacity", 0.55);
    const nodeLayer  = root.append("g");
    const labelLayer = root.append("g");

    const link = linkLayer.selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", 1.25)
      .attr("marker-end", "url(#arrow)");

    const nodeRadius  = 5;
    const hoverRadius = 7;
    const gapOffset   = 4; // reverted modest gap

    const nodeFill = (d) => (data.degree.get(String(d.id)) === data.maxDeg ? ORANGE : BLACK);

    let sim; // for drag reheat

    const node = nodeLayer.selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", nodeRadius)
      .attr("fill", nodeFill)
      .attr("stroke", "white")
      .attr("stroke-width", 1.6)
      .style("cursor", "pointer")
      .on("click", (_, d) => onSelect && onSelect(d))
      .on("mouseover", function (_, d) {
        d3.select(this).transition().duration(120).attr("r", hoverRadius).attr("fill", GREEN);
        const id = String(d.id);
        link.attr("stroke-opacity", L =>
          String(L.source.id ?? L.source) === id || String(L.target.id ?? L.target) === id ? 0.9 : 0.15
        );
        node.attr("opacity", n => (n === d ? 1 : 0.35));
        label.filter(ld => ld === d).attr("opacity", 1);
      })
      .on("mouseout", function () {
        d3.select(this).transition().duration(120).attr("r", nodeRadius).attr("fill", d => nodeFill(d));
        link.attr("stroke-opacity", 0.55);
        node.attr("opacity", 1);
        label.attr("opacity", 0);
      })
      .call(
        d3.drag()
          .on("start", (event, d) => { if (sim) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag",  (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end",   (event, d) => { d.fx = null; d.fy = null; if (sim) sim.alphaTarget(0); })
      );

    // labels: text with white halo (reverted)
    const labelText = d => d.title ? (d.title.length > 42 ? d.title.slice(0, 41) + "…" : d.title) : String(d.id);
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

    // --- pan & zoom (shared instance) – normal cursor; grabbing only while dragging
    const zoom = d3.zoom()
      .scaleExtent([0.3, 6])
      .on("start", (e) => svg.style("cursor", "grabbing"))
      .on("end",   (e) => svg.style("cursor", "default"))
      .on("zoom",  (e) => root.attr("transform", e.transform));
    zoomRef.current = zoom;
    svg.call(zoom).style("cursor", "default");

    // angular-spread (gentle)
    function forceAngularSpread(minSep = 0.35, strength = 0.03) {
      let nodeById, adj;
      function init() {
        nodeById = new Map(nodes.map(n => [String(n.id), n]));
        adj = new Map(nodes.map(n => [n, []]));
        for (const l of links) {
          const s = nodeById.get(String(l.source.id ?? l.source));
          const t = nodeById.get(String(l.target.id ?? l.target));
          if (s && t) { adj.get(s).push(t); adj.get(t).push(s); }
        }
      }
      function force() {
        for (const v of nodes) {
          const neigh = adj.get(v); if (!neigh || neigh.length < 2) continue;
          const items = neigh.map(n => {
            const dx = (n.x ?? 0) - (v.x ?? 0);
            const dy = (n.y ?? 0) - (v.y ?? 0);
            const a = Math.atan2(dy, dx);
            return { n, a, dx, dy };
          }).sort((a,b)=>a.a-b.a);
          for (let i=0;i<items.length;i++){
            const A=items[i], B=items[(i+1)%items.length];
            let delta=B.a-A.a; if (delta<=0) delta+=Math.PI*2;
            if (delta<minSep){
              const push=(minSep-delta)*strength;
              const ra=Math.hypot(A.dx,A.dy)||1, rb=Math.hypot(B.dx,B.dy)||1;
              const tax=(A.dy/ra), tay=-(A.dx/ra);
              const tbx=-(B.dy/rb), tby=(B.dx/rb);
              A.n.vx=(A.n.vx||0)+tax*push;  A.n.vy=(A.n.vy||0)+tay*push;
              B.n.vx=(B.n.vx||0)+tbx*push;  B.n.vy=(B.n.vy||0)+tby*push;
              v.vx=(v.vx||0)-(tax+tbx)*(push*0.5);
              v.vy=(v.vy||0)-(tay+tby)*(push*0.5);
            }
          }
        }
      }
      force.initialize = init;
      return force;
    }

    // forces
    sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d=>String(d.id)).distance(70).strength(0.55))
      .force("charge", d3.forceManyBody().strength(-320))
      .force("center", d3.forceCenter(0,0))
      .force("collision", d3.forceCollide().radius(12).iterations(2))
      .force("angular", forceAngularSpread(0.35, 0.03))
      .velocityDecay(0.25)
      .alpha(1)
      .alphaDecay(0.06);

    // small trim at both ends; keep modest gap; arrow covers line tip visually
    function endpointWithGap(d, which) {
      const sx=d.source.x, sy=d.source.y, tx=d.target.x, ty=d.target.y;
      const dx=tx-sx, dy=ty-sy, dist=Math.hypot(dx,dy)||1, ux=dx/dist, uy=dy/dist;
      const srcR=nodeRadius+gapOffset, tarR=nodeRadius+gapOffset+1.2;
      if (which==="x1") return sx + ux*srcR;
      if (which==="y1") return sy + uy*srcR;
      if (which==="x2") return tx - ux*tarR;
      if (which==="y2") return ty - uy*tarR;
    }

    sim.on("tick", () => {
      link
        .attr("x1", d => endpointWithGap(d,"x1"))
        .attr("y1", d => endpointWithGap(d,"y1"))
        .attr("x2", d => endpointWithGap(d,"x2"))
        .attr("y2", d => endpointWithGap(d,"y2"));
      node.attr("cx", d=>d.x).attr("cy", d=>d.y);
      label.attr("x", d=>(d.x??0)+9).attr("y", d=>(d.y??0)+4);
    });

    // initial fit — use the SAME zoom instance → no click needed
    immediateFit(svg, zoom, nodes, width, height, 32);

    // handle future fits (Refresh/Focus)
    const onFit = () => animateFit(svg, zoom, nodes, width, height, 32);
    svg.on("smartfit", onFit);

    return () => {
      svg.on("smartfit", null);
      sim.stop();
    };
  }, [data]);

  // trigger fit on demand
  useEffect(() => {
    d3.select(svgRef.current).dispatch("smartfit");
  }, [fitSignal]);

  return (
    <svg ref={svgRef} className={`w-full h-full rounded-lg bg-white ${className}`}>
      <g className="root" />
    </svg>
  );
}

/* ----- fitting helpers that use the SAME zoom instance ----- */

function computeFit(nodes, width, height, pad=32) {
  const xs = nodes.map(n => n.x||0), ys = nodes.map(n => n.y||0);
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  const w=Math.max(1, maxX-minX), h=Math.max(1, maxY-minY);
  const k = Math.min(1, 0.9 / Math.max(w/(width-pad*2), h/(height-pad*2)));
  const cx=(minX+maxX)/2, cy=(minY+maxY)/2;
  const tx = width/2 - k*cx, ty = height/2 - k*cy;
  return d3.zoomIdentity.translate(tx, ty).scale(k);
}

function immediateFit(svg, zoom, nodes, width, height, pad=32) {
  const t = computeFit(nodes, width, height, pad);
  svg.call(zoom.transform, t); // IMPORTANT: same zoom instance
}

function animateFit(svg, zoom, nodes, width, height, pad=32) {
  const t0 = d3.zoomTransform(svg.node());
  const t  = computeFit(nodes, width, height, pad);
  const dist = Math.hypot(t0.x - t.x, t0.y - t.y) + Math.abs(t0.k - t.k)*50;
  if (dist < 20) return; // already good enough
  svg.transition().duration(500).call(zoom.transform, t);
}
