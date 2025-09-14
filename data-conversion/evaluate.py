import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import networkx as nx
import csv
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# Crossref helpers
# -------------------------
def fetch_doi_data(doi):
    url = f"https://api.crossref.org/works/{doi}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()['message']
    except Exception:
        return None
    return None

def get_references_from_doi_parallel(dois, max_workers=8):
    refs_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doi = {executor.submit(fetch_doi_data, doi): doi for doi in dois}
        for future in tqdm(as_completed(future_to_doi), total=len(future_to_doi), desc="Fetching DOIs"):
            doi = future_to_doi[future]
            data = future.result()
            refs_map[doi] = []
            if data and 'reference' in data:
                for ref in data['reference']:
                    ref_doi = ref.get('DOI')
                    if ref_doi:
                        refs_map[doi].append(ref_doi)
    return refs_map

# -------------------------
# Cache for total citations to avoid repeated API calls
# -------------------------
total_citations_cache = {}

def get_total_citations(doi):
    if doi in total_citations_cache:
        return total_citations_cache[doi]
    data = fetch_doi_data(doi)
    count = 1
    if data:
        count = data.get('reference-count', 1)
    total_citations_cache[doi] = count
    return count

# -------------------------
# Convert NetworkX graph to PyG Data
# -------------------------
def pyg_data_from_nx(G):
    doi_list = list(G.nodes)
    node_features = []
    for node in doi_list:
        data_node = G.nodes[node]
        total_cites = get_total_citations(node)
        node_features.append([
            data_node['layer_first_seen'],
            data_node['indegree'],
            data_node['direct'],
            data_node['freq'],
            total_cites  # fifth feature
        ])
    x = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_index = []
    for u, v in G.edges:
        edge_index.append([doi_list.index(u), doi_list.index(v)])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
    return Data(x=x, edge_index=edge_index), doi_list

# -------------------------
# GraphSAGE model
# -------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        if x.size(0) < 2 or edge_index.size(1) == 0:
            return torch.zeros((x.size(0), self.conv2.out_channels), device=x.device)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# -------------------------
# Layer expansion with pruning
# -------------------------
def add_layer_with_pruning(G, current_layer, refs_map):
    new_nodes = []
    nodes_current_layer = [n for n, d in G.nodes(data=True) if d['layer_first_seen'] == current_layer - 1]

    for node in tqdm(nodes_current_layer, desc=f"Layer {current_layer}", unit="node"):
        references = refs_map.get(node, [])
        for ref in references:
            if ref not in G:
                direct_flag = 1 if current_layer == 1 else 0
                G.add_node(ref, layer_first_seen=current_layer, freq=1, direct=direct_flag, indegree=0)
                new_nodes.append(ref)
            else:
                G.nodes[ref]['freq'] += 1
            G.add_edge(node, ref)

    # Update indegree
    for n in G.nodes:
        G.nodes[n]['indegree'] = G.in_degree(n)

    # Pruning rules
    to_remove = []
    for n in list(G.nodes):
        layer_seen = G.nodes[n]['layer_first_seen']
        freq = G.nodes[n]['freq']
        total_cites = get_total_citations(n)

        if layer_seen >= 2:
            appears_next = any(n in refs_map.get(parent, []) 
                               for parent in G.nodes 
                               if G.nodes[parent]['layer_first_seen'] == layer_seen)
            if not appears_next or total_cites < 50:
                to_remove.append(n)

    for n in to_remove:
        if n in G:
            G.remove_node(n)

    return G, new_nodes

# -------------------------
# Adaptive graph builder
# -------------------------
def build_graph_adaptive(seed_doi, model, max_layers=10, embed_change_threshold=1e-3):
    G = nx.DiGraph()
    G.add_node(seed_doi, layer_first_seen=0, freq=1, direct=0, indegree=0)

    previous_embeddings = None
    current_layer_nodes = [seed_doi]
    refs_map = {}

    for layer in range(1, max_layers + 1):
        new_refs_map = get_references_from_doi_parallel(current_layer_nodes)
        refs_map.update(new_refs_map)

        G, new_nodes = add_layer_with_pruning(G, layer, refs_map)
        if not new_nodes:
            if layer == 1:
                return None
            print(f"No new nodes added at layer {layer}, stopping expansion.")
            break

        current_layer_nodes = new_nodes

        # Check embedding convergence
        data, doi_list = pyg_data_from_nx(G)
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)
        if previous_embeddings:
            common_nodes = set(previous_embeddings.keys()).intersection(doi_list)
            if common_nodes:
                diffs = [(embeddings[doi_list.index(n)] - previous_embeddings[n]).norm().item() for n in common_nodes]
                if np.mean(diffs) < embed_change_threshold:
                    break
        previous_embeddings = {n: embeddings[doi_list.index(n)] for n in doi_list}

    return G

# -------------------------
# Helper: evaluate DOI
# -------------------------
def evaluate_doi(model, doi, max_layers=10, top_k=3, output_csv=None):
    G = build_graph_adaptive(doi, model, max_layers=max_layers)
    if not G:
        print(f"DOI {doi} did not expand enough to build a graph.")
        return None

    data, doi_list = pyg_data_from_nx(G)
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        scores = embeddings.norm(dim=1).cpu().numpy()

    for node, score in zip(doi_list, scores):
        G.nodes[node]['score'] = score

    ranked = sorted(G.nodes(data=True), key=lambda x: x[1]['score'], reverse=True)

    if output_csv:
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["DOI","Layer","Indegree","Direct","Frequency","TotalCites","Score"])
            for node, data_node in G.nodes(data=True):
                total_cites = get_total_citations(node)
                writer.writerow([
                    node,
                    data_node['layer_first_seen'],
                    data_node['indegree'],
                    data_node['direct'],
                    data_node['freq'],
                    total_cites,
                    round(data_node['score'], 4)
                ])

    print(f"\nTop {top_k} recommended articles for DOI {doi}:")
    for node, data_node in ranked[:top_k]:
        print(f"DOI: {node} | Layer: {data_node['layer_first_seen']} "
              f"| Score: {data_node['score']:.4f} "
              f"| Direct: {data_node['direct']} | Frequency: {data_node['freq']}")

    return ranked

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Load saved model (in_channels=5 due to total_cites feature)
    model = GraphSAGE(in_channels=5, hidden_channels=16, out_channels=8).to(device)
    model.load_state_dict(torch.load("/mnt/c/Users/oliwi/HopHacks2025/gnn_citation_model.pt", map_location=device))
    model.eval()

    # Test DOI evaluation
    test_doi = "10.1021/nl5049753"
    evaluate_doi(model, test_doi, max_layers=8, top_k=5, output_csv="results.csv")
