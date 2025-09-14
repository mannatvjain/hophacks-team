# backend/main.py
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import ast
import re

# ---------- API contract ----------
class GraphRequest(BaseModel):
    doi: str
    depth: int | None = None   # NEW

class Node(BaseModel):
    id: str
    title: str | None = None
    year: int | None = None
    authors: List[str] | None = None
    outCitations: List[str] = []
    score: float | int | None = None
    abstract: str | None = None

class Link(BaseModel):
    source: str
    target: str

class GraphResponse(BaseModel):
    nodes: List[Node]
    links: List[Link]
    shortest_distance: List[str]
    description: List[Tuple[str, str]]

# ---------- app ----------
app = FastAPI()

# Allow your Vite dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ---------- dataset load (once) ----------
CSV_PATH = "data/final_merged_with_abstracts.csv"

def _norm_doi(x) -> str | None:
    if isinstance(x, str):
        s = x.strip()
        return s.lower() if s else None
    return None

def _parse_refs(x) -> List[str]:
    if not isinstance(x, str):
        return []
    try:
        v = ast.literal_eval(x)
    except Exception:
        return []
    if isinstance(v, (list, tuple, set)):
        out = []
        for item in v:
            d = _norm_doi(item)
            if isinstance(item, str) and d:
                out.append(d)
        return out
    return []

_df = pd.read_csv(CSV_PATH)

# Universe of DOIs present in the dataset (normalized)
_ID_SET = { _norm_doi(d) for d in _df["DOI"].dropna().tolist() }
_ID_SET.discard(None)

def _get_str(row, *keys) -> str | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s
    return None

# Build nodes
def _strip_angle_tags(s: str | None) -> str | None:
    if not isinstance(s, str):
        return s
    # remove anything like <tag ...> ... </tag> or standalone <.../>
    s = re.sub(r"<[^>]*>", "", s)
    # collapse whitespace and trim
    return re.sub(r"\s+", " ", s).strip()

_ALL_NODES: List[Dict[str, Any]] = []
for _, row in _df.iterrows():
    doi = _norm_doi(row.get("DOI"))
    if not doi:
        continue

    refs = _parse_refs(row.get("References"))
    # keep internal edges only
    refs = [r for r in refs if r in _ID_SET]

    authors_raw = _get_str(row, "Authors", "authors")
    authors = [a.strip() for a in authors_raw.split(";")] if authors_raw else None

    year_val = row.get("Year")
    year = int(year_val) if pd.notna(year_val) else None

    score_val = row.get("Score")
    score = float(score_val) if pd.notna(score_val) else None

    title = _get_str(row, "Title", "title")
    abstract = _strip_angle_tags(_get_str(row, "Abstract", "abstract"))

    _ALL_NODES.append({
        "id": doi,
        "title": title if isinstance(title, str) else None,
        "year": year,
        "authors": authors,
        "outCitations": refs,
        "score": score,
        "abstract": abstract if isinstance(abstract, str) else None,
    })

# Build links (source -> target) from outCitations
_ALL_LINKS: List[Dict[str, str]] = [
    {"source": n["id"], "target": t}
    for n in _ALL_NODES
    for t in n.get("outCitations", [])
]

from collections import defaultdict

# Undirected adjacency for hop counting
_ADJ = defaultdict(set)
for e in _ALL_LINKS:
    s, t = e["source"], e["target"]
    _ADJ[s].add(t); _ADJ[t].add(s)

def _gold_id_fallback() -> str:
    # pick highest-score node, else first node
    scored = [n for n in _ALL_NODES if isinstance(n.get("score"), (int, float))]
    return (max(scored, key=lambda n: n["score"]) if scored else _ALL_NODES[0])["id"]

def _bfs_ids(start_id: str | None, depth: int = 2, max_nodes: int = 600) -> set[str]:
    start = _norm_doi(start_id)
    if not start or start not in _ID_SET:
        start = _gold_id_fallback()
    keep = {start}
    frontier = [start]
    for _ in range(depth):
        nxt = []
        for u in frontier:
            for v in _ADJ.get(u, ()):
                if v not in keep:
                    keep.add(v); nxt.append(v)
                    if len(keep) >= max_nodes:
                        break
            if len(keep) >= max_nodes:
                break
        frontier = nxt
        if not frontier or len(keep) >= max_nodes:
            break
    return keep

# Curated per-DOI descriptions (pairs). Keep lowercased DOIs.
_DESCRIPTION: List[Tuple[str, str]] = [
    # ("10.1016/j.tins.2010.01.006", "description for path node 1"),
    # ("10.1016/j.cell.2020.12.015", "description for background study B"),
]

# ---------- logic ----------
def build_graph_for_doi(doi: str, depth: int | None = None) -> Dict[str, Any]:
    d = depth if isinstance(depth, int) and depth >= 0 else 2
    keep = _bfs_ids(doi, depth=d, max_nodes=600)
    nodes = [n for n in _ALL_NODES if n["id"] in keep]
    links = [e for e in _ALL_LINKS if e["source"] in keep and e["target"] in keep]
    return {
        "nodes": nodes,
        "links": links,
        "shortest_distance": [],
        "description": _DESCRIPTION,
    }

# ---------- endpoint ----------
@app.post("/api/graph", response_model=GraphResponse)
async def api_graph(req: GraphRequest) -> GraphResponse:
    data = build_graph_for_doi(req.doi, req.depth)  # NEW
    return data
