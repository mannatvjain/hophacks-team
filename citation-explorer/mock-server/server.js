// mock-server/server.js
import express from "express";

const app = express();
app.use(express.json());

// Dummy graph
const demo = {
  nodes: [
    {
      id: "10.1038/s41586-020-03167-3",
      score: 3,
      title: "Original Research Paper",
      year: 2021,
      authors: ["Smith J", "Lee K"],
      outCitations: ["10.1126/science.aaz1776", "10.1016/j.cell.2020.12.015"],
    },
    { id: "10.1126/science.aaz1776", score: 1, title: "Background Study A", year: 2019, authors: ["Patel R"], outCitations: [] },
    { id: "10.1016/j.cell.2020.12.015", score: 1, title: "Background Study B", year: 2020, authors: ["Chen X","Ng M"], outCitations: [] },
  ],
};

function withLinks(data) {
  const nodes = (data.nodes ?? []).map(d => ({ ...d, id: String(d.id) }));
  const idSet = new Set(nodes.map(n => n.id));
  const links = [];
  for (const s of nodes) {
    for (const t of (s.outCitations || [])) {
      const tid = String(t);
      if (idSet.has(tid)) links.push({ source: String(s.id), target: tid });
    }
  }
  return { nodes, links };
}

app.post("/api/graph", (req, res) => {
  const { doi } = req.body || {};
  console.log("gopgop:", doi);
  res.json(withLinks(demo));
});

const PORT = process.env.MOCK_PORT || 8787;
app.listen(PORT, () => console.log(`Mock server on http://localhost:${PORT}`));
