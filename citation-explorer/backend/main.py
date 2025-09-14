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
    abstract: str | None = None  # the abstract (doy)

class Link(BaseModel):
    source: str
    target: str

class GraphResponse(BaseModel):
    nodes: List[Node]
    links: List[Link]
    shortest_distance: List[str]  # <-- NEW

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
            "id": "10.1038/nature14236",  # origin
            "title": "Origin Paper",
            "year": 2015,
            "authors": ["Alpha A"],
            "outCitations": ["10.1016/j.tins.2010.01.006", "10.1126/science.aaz1776"],
            "score": 5,
            "abstract": "Demo abstract for origin."
        },
        {
            "id": "10.1016/j.tins.2010.01.006",  # path 1
            "title": "Path Node 1",
            "year": 2010,
            "authors": ["Bravo B"],
            "outCitations": ["10.1016/0306-4522(89)90423-5"],
            "score": 3,
            "abstract": "Demo abstract for path node 1."
        },
        {
            "id": "10.1016/0306-4522(89)90423-5",  # path 2
            "title": "Path Node 2",
            "year": 1989,
            "authors": ["Charlie C"],
            "outCitations": ["10.1016/0304-3940(86)90466-0"],
            "score": 2,
            "abstract": "Demo abstract for path node 2."
        },
        {
            "id": "10.1016/0304-3940(86)90466-0",  # endpoint
            "title": "Endpoint Paper",
            "year": 1986,
            "authors": ["Delta D"],
            "outCitations": [],
            "score": 4,
            "abstract": "Demo abstract for endpoint."
        },
        {
            "id": "10.1126/science.aaz1776",  # distractor branch
            "title": "Background Study A",
            "year": 2019,
            "authors": ["Patel R"],
            "outCitations": ["10.1016/0304-3940(86)90466-0"],  # jumps to endpoint (non-path)
            "score": 1,
            "abstract": "Demo abstract A."
        },
        {
            "id": "10.1016/j.cell.2020.12.015",  # unrelated
            "title": "Background Study B",
            "year": 2020,
            "authors": ["Chen X", "Ng M"],
            "outCitations": [],
            "score": 1,
            "abstract": "Demo abstract B."
        },
    ]
}
    # derive links from outCitations (same as mock)
    ids = {n["id"] for n in demo["nodes"]}
    links = [{"source": n["id"], "target": t} for n in demo["nodes"] for t in n.get("outCitations", []) if t in ids]
    return {
    "nodes": demo["nodes"],
    "links": links,
    # two-id dummy path (origin -> endpoint); replace later with your CSV array
    "shortest_distance": [
    "10.1038/nature14236",
    "10.1016/j.tins.2010.01.006",
    "10.1016/0306-4522(89)90423-5",
    "10.1016/0304-3940(86)90466-0",
],
}


# ---------- endpoint ----------
@app.post("/api/graph", response_model=GraphResponse)
async def api_graph(req: GraphRequest) -> GraphResponse:
    data = build_graph_for_doi(req.doi)
    return data  # FastAPI will validate/serialize to the contract
