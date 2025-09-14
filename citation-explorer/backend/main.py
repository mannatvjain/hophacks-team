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
                "abstract": "Predicting protein properties is paramount for biological and medical advancements. Current protein engineering mutates on a typical protein, called the wild-type, to construct a family of homologous proteins and study their properties. Yet, existing methods easily neglect subtle mutations, failing to capture the effect on the protein properties. To this end, we propose EvolMPNN, Evolution-aware Message Passing Neural Network, an efficient model to learn evolution-aware protein embeddings. EvolMPNN samples sets of anchor proteins, computes evolutionary information by means of residues and employs a differentiable evolution-aware aggregation scheme over these sampled anchors. This way, EvolMPNN can efficiently utilise a novel message-passing method to capture the mutation effect on proteins with respect to the anchor proteins. Afterwards, the aggregated evolution-aware embeddings are integrated with sequence embeddings to generate final comprehensive protein embeddings. Our model shows up to 6.4 better than state-of-the-art methods and attains 36X inference speedup in comparison with large pre-trained models. Code and models are available at this https URL.Predicting protein properties is paramount for biological and medical advancements. Current protein engineering mutates on a typical protein, called the wild-type, to construct a family of homologous proteins and study their properties. Yet, existing methods easily neglect subtle mutations, failing to capture the effect on the protein properties. To this end, we propose EvolMPNN, Evolution-aware Message Passing Neural Network, an efficient model to learn evolution-aware protein embeddings. EvolMPNN samples sets of anchor proteins, computes evolutionary information by means of residues and employs a differentiable evolution-aware aggregation scheme over these sampled anchors. This way, EvolMPNN can efficiently utilise a novel message-passing method to capture the mutation effect on proteins with respect to the anchor proteins. Afterwards, the aggregated evolution-aware embeddings are integrated with sequence embeddings to generate final comprehensive protein embeddings. Our model shows up to 6.4 better than state-of-the-art methods and attains 36X inference speedup in comparison with large pre-trained models. Code and models are available at this https URL."
            },
            {"id": "10.1126/science.aaz1776", "title": "Background Study A", "year": 2019, "authors": ["Patel R"], "outCitations": [], "score": 1, "abstract": "This is a demo abstract for background study A."},
            {"id": "10.1016/j.cell.2020.12.015", "title": "Background Study B", "year": 2020, "authors": ["Chen X", "Ng M"], "outCitations": [], "score": 1, "abstract": "This is a demo abstract for background study B."},
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
