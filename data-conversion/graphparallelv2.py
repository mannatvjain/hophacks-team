# ------------------------
# Requirements:
# pip install torch torch-geometric networkx matplotlib tqdm requests
# ------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import csv
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from collections import defaultdict
from tqdm import tqdm
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import numpy as np

# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------
# Crossref API helpers + cache
# ------------------------
doi_cache = {}

def fetch_doi_data(doi):
    doi = doi.lower()
    if doi in doi_cache:
        return doi_cache[doi]
    url = f"https://api.crossref.org/works/{doi}"
    headers = {'User-Agent': 'CitationGraph/1.0 (mailto:your-email@example.com)'}
    try:
        time.sleep(random.uniform(0.05, 0.25))
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json().get('message', None)
            doi_cache[doi] = data
            return data
        elif r.status_code == 404:
            print(f"DOI {doi} not found (404)")
        else:
            print(f"Error fetching {doi}: HTTP {r.status_code}")
    except Exception as e:
        print(f"Exception fetching {doi}: {str(e)}")
    doi_cache[doi] = None
    return None

def get_references(doi):
    doi = doi.lower()
    data = fetch_doi_data(doi)
    if data and 'reference' in data:
        refs = []
        for ref in data['reference']:
            rdoi = ref.get('DOI')
            if rdoi:
                refs.append(rdoi.lower())
        return refs
    return []

def get_citation_count(doi):
    doi = doi.lower()
    data = fetch_doi_data(doi)
    if data:
        return int(data.get('is-referenced-by-count', 0) or 0)
    return 0

# ------------------------
# Parallel fetching helpers
# ------------------------
def fetch_references_parallel(dois, max_workers=4):
    refs_map = {}
    def worker(doi):
        time.sleep(random.uniform(0.05, 0.2))
        return doi, get_references(doi)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, doi): doi for doi in dois}
        for future in tqdm(as_completed(futures), total=len(dois), desc="Fetching references"):
            doi = futures[future]
            try:
                doi_res, refs = future.result()
                refs_map[doi_res] = refs
            except Exception as e:
                print(f"Error fetching refs for {doi}: {e}")
                refs_map[doi] = []
    return refs_map

def fetch_citation_counts_parallel(dois, max_workers=4):
    counts_map = {}
    def worker(doi):
        time.sleep(random.uniform(0.02, 0.12))
        return doi, get_citation_count(doi)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, doi): doi for doi in dois}
        for future in tqdm(as_completed(futures), total=len(dois), desc="Fetching citation counts"):
            doi = futures[future]
            try:
                doi_res, cnt = future.result()
                counts_map[doi_res] = cnt
            except Exception as e:
                print(f"Error fetching citation count for {doi}: {e}")
                counts_map[doi] = 0
    return counts_map

# ------------------------
# Graph expansion
# ------------------------
def add_layer_parallel(G, current_layer, max_refs_per_node=20, max_workers=4):
    new_nodes = []
    layer_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == current_layer - 1]
    if not layer_nodes:
        return G, new_nodes

    references_dict = fetch_references_parallel(layer_nodes, max_workers=max_workers)

    all_new_dois = set()
    for refs in references_dict.values():
        for r in refs:
            if r not in G:
                all_new_dois.add(r)
    all_new_dois = list(all_new_dois)
    citation_counts = fetch_citation_counts_parallel(all_new_dois, max_workers=max_workers)

    for node in tqdm(layer_nodes, desc=f"Processing nodes in layer {current_layer}"):
        refs = references_dict.get(node, [])[:max_refs_per_node]
        for ref in refs:
            if ref not in G:
                direct_flag = 1 if current_layer == 1 else 0
                citation_count = citation_counts.get(ref, get_citation_count(ref))
                G.add_node(ref,
                           layer=current_layer,
                           indegree=1,
                           direct=direct_flag,
                           citation_count=citation_count,
                           freq=1)
                new_nodes.append(ref)
            else:
                G.nodes[ref]['freq'] = G.nodes[ref].get('freq', 1) + 1
                G.nodes[ref]['indegree'] = G.in_degree(ref)
            G.add_edge(node, ref)

    # Pruning
    current_layer_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == current_layer]
    to_remove = []
    for n in tqdm(current_layer_nodes, desc=f"Pruning layer {current_layer}"):
        if G.nodes[n]['layer'] == 0 or G.nodes[n].get('direct', 0) == 1:
            continue
        if current_layer >= 2:
            citing_papers = [u for u, v in G.in_edges(n) if G.nodes[u]['layer'] < current_layer]
            if len(citing_papers) < 2 and G.nodes[n].get('citation_count', 0) < 10:
                to_remove.append(n)

    for n in to_remove:
        if n in G:
            G.remove_node(n)
            if n in new_nodes:
                new_nodes.remove(n)

    for n in tqdm(G.nodes, desc="Updating indegree"):
        G.nodes[n]['indegree'] = G.in_degree(n)

    return G, new_nodes

def build_initial_graph(source):
    G = nx.DiGraph()
    source = source.lower()
    citation_count = get_citation_count(source)
    G.add_node(source, layer=0, indegree=0, direct=0, citation_count=citation_count, freq=1)
    return G

# ------------------------
# Normalize helper
# ------------------------
def normalize_list(values):
    if len(values) == 0:
        return values
    arr = np.array(values, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx == mn:
        return [0.0 for _ in values]
    return ((arr - mn) / (mx - mn)).tolist()

# ------------------------
# Convert NX -> PyG
# ------------------------
def pyg_data_from_nx(G):
    doi_list = list(G.nodes)
    layers, indegrees, directs, citations = [], [], [], []

    for node in doi_list:
        d = G.nodes[node]
        if d.get('layer', 0) == 0:
            layers.append(0.0)
            indegrees.append(0.0)
            directs.append(0.0)
            citations.append(0.0)
        else:
            layers.append(1.0 / float(d.get('layer', 1)))
            indegrees.append(float(d.get('indegree', 0)))
            directs.append(float(d.get('direct', 0)))
            citations.append(float(d.get('citation_count', 0)))

    indegrees_norm = [x * 4 for x in normalize_list(indegrees)]
    directs_norm = normalize_list(directs)
    citations_norm = normalize_list(citations)

    node_features = [[layers[i], indegrees_norm[i], directs_norm[i], citations_norm[i]] for i in range(len(doi_list))]
    x = torch.tensor(node_features, dtype=torch.float).to(device)

    edge_index = []
    for u, v in G.edges:
        edge_index.append([doi_list.index(u), doi_list.index(v)])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

    return Data(x=x, edge_index=edge_index), doi_list

# ------------------------
# GraphSAGE
# ------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# ------------------------
# Build citation graph
# ------------------------
def build_citation_graph(source_doi, max_layers=3, top_k=3, epochs_per_layer=50, max_refs_per_node=20):
    source_doi = source_doi.lower()
    source_data = fetch_doi_data(source_doi)
    if source_data is None:
        print(f"Source DOI {source_doi} not found")
        return None

    G = build_initial_graph(source_doi)
    current_layer = 1
    previous_top = []
    converged = False

    in_features = 4
    model = GraphSAGE(in_channels=in_features, hidden_channels=32, out_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    pbar = tqdm(total=max_layers, desc="Building citation graph layers")
    while current_layer <= max_layers and not converged:
        G, new_nodes = add_layer_parallel(G, current_layer, max_refs_per_node=max_refs_per_node)
        if not new_nodes:
            pbar.update(max_layers - current_layer + 1)
            break

        data, doi_list = pyg_data_from_nx(G)
        data.x += 0.01 * torch.randn_like(data.x)

        model.train()
        for epoch in tqdm(range(epochs_per_layer), desc=f"Training GNN for layer {current_layer}"):
            optimizer.zero_grad()
            embeddings = model(data.x, data.edge_index)
            if data.edge_index.size(1) > 0:
                loss_terms = []
                for u, v in zip(data.edge_index[0], data.edge_index[1]):
                    weight = 1.0 + G.nodes[doi_list[v.item()]]['citation_count'] / 10.0
                    loss_terms.append(F.mse_loss(embeddings[u], embeddings[v]) * weight)
                loss = torch.mean(torch.stack(loss_terms))
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)
            if data.edge_index.size(1) > 0:
                scores = embeddings.norm(dim=1).cpu().numpy()
            else:
                scores = np.ones(len(doi_list), dtype=float)
            for node, sc in zip(doi_list, scores):
                G.nodes[node]['score'] = float(sc)

        ranked = sorted(G.nodes(data=True), key=lambda x: x[1].get('score', 0), reverse=True)
        current_top = [node for node, _ in ranked[:top_k]]
        if previous_top and current_top == previous_top:
            print(f"Top-{top_k} converged at layer {current_layer}")
            converged = True
            pbar.update(max_layers - current_layer + 1)
            break
        previous_top = current_top

        pbar.update(1)
        current_layer += 1

    pbar.close()
    return G

# ------------------------
# Save edges to CSV
# ------------------------
def save_edges_to_csv(G, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Citing_DOI", "Cited_DOI", "Citing_Layer", "Cited_Layer", "Citation_Type"])
        for u, v in tqdm(G.edges(), desc="Saving edges to CSV"):
            citing_layer = G.nodes[u].get('layer', '')
            cited_layer = G.nodes[v].get('layer', '')
            citation_type = "Direct" if G.nodes[v].get('direct', 0) == 1 else "Indirect"
            writer.writerow([u, v, citing_layer, cited_layer, citation_type])
    print(f"Edges saved to {filename}")

# ------------------------
# Main run example
# ------------------------
if __name__ == "__main__":
    test_dois = [
        "10.1038/nature14236"
    ]

    for source_doi in test_dois:
        print(f"\nBuilding citation graph for DOI: {source_doi}")
        G = build_citation_graph(source_doi, max_layers=3, top_k=3, epochs_per_layer=40, max_refs_per_node=20)
        if G is None:
            continue

        safe_name = source_doi.replace("/", "_")
        # Save node scores
        with open(f"citation_scores_{safe_name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["DOI", "Layer", "Indegree", "Direct", "Freq", "Citation_Count", "Score"])
            for node, data_node in tqdm(G.nodes(data=True), desc="Saving node scores"):
                writer.writerow([
                    node,
                    data_node.get('layer', ''),
                    data_node.get('indegree', 0),
                    data_node.get('direct', 0),
                    data_node.get('freq', 0),
                    data_node.get('citation_count', 0),
                    round(data_node.get('score', 0), 6)
                ])
        print("Node scores saved.")
        save_edges_to_csv(G, f"citation_edges_{safe_name}.csv")
