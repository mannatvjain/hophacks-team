# backend/main.py
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ---------- API contract ----------
class GraphRequest(BaseModel):
    doi: str

class Node(BaseModel):
    id: str
    title: str | None = None
    year: int | None = None
    authors: List[str] | None = None
    outCitations: List[str] = []
    score: float | int | None = None  # optional; used by frontend

class Link(BaseModel):
    source: str
    target: str

class GraphResponse(BaseModel):
    nodes: List[Node]
    links: List[Link]

# ---------- app ----------
app = FastAPI()

# Allow your Vite dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ---------- paste your real logic here ----------
def build_graph_for_doi(doi: str) -> Dict[str, Any]:
    """
    TODO: Replace this with your real Python pipeline:
      - fetch metadata for `doi`
      - fetch/scrape references/citations
      - construct nodes (with optional `score`) and links
    Return shape: { "nodes": [...], "links": [...] }
    """
    demo = {
        "nodes": [
            {
                "id": "10.1038/s41586-020-03167-3",
                "title": "Original Research Paper",
                "year": 2021,
                "authors": ["Smith J", "Lee K"],
                "outCitations": ["10.1126/science.aaz1776", "10.1016/j.cell.2020.12.015"],
                "score": 3,
            },
            {"id": "10.1126/science.aaz1776", "title": "Background Study A", "year": 2019, "authors": ["Patel R"], "outCitations": [], "score": 1},
            {"id": "10.1016/j.cell.2020.12.015", "title": "Background Study B", "year": 2020, "authors": ["Chen X", "Ng M"], "outCitations": [], "score": 1},
        ]
    }
    # derive links from outCitations (same as mock)
    ids = {n["id"] for n in demo["nodes"]}
    links = [{"source": n["id"], "target": t} for n in demo["nodes"] for t in n.get("outCitations", []) if t in ids]
    return {"nodes": demo["nodes"], "links": links}

# ---------- endpoint ----------
@app.post("/api/graph", response_model=GraphResponse)
async def api_graph(req: GraphRequest) -> GraphResponse:
    data = build_graph_for_doi(req.doi)
    return data  # FastAPI will validate/serialize to the contract
