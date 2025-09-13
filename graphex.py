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
import numpy as np

# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------
# CrossRef helpers
# ------------------------
doi_cache = {}

def fetch_doi_data(doi):
    if doi in doi_cache:
        return doi_cache[doi]
    
    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        'User-Agent': 'CitationGraph/1.0 (mailto:your-email@example.com)'
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()['message']
            doi_cache[doi] = data
            return data
        else:
            print(f"Error fetching {doi}: HTTP {r.status_code}")
    except Exception as e:
        print(f"Exception fetching {doi}: {str(e)}")
    
    doi_cache[doi] = None
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
                        refs_map[doi].append(ref_doi.lower())
    return refs_map

def get_total_citations(doi):
    data = fetch_doi_data(doi)
    if data:
        return data.get('is-referenced-by-count', 1)
    return 1

# ------------------------
# Convert NetworkX graph to PyG Data
# ------------------------
def pyg_data_from_nx(G):
    doi_list = list(G.nodes)
    node_features = []
    for node in doi_list:
        data = G.nodes[node]
        node_features.append([
            data['layer_first_seen'],
            data['indegree'],
            data['direct'],
            data['freq'],
            data.get('citation_count', 1)  # Add citation count as feature
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

# ------------------------
# GraphSAGE model
# ------------------------
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

# ------------------------
# Layer expansion with pruning
# ------------------------
def add_layer_with_pruning(G, current_layer, refs_map):
    new_nodes = []
    nodes_current_layer = [n for n, d in G.nodes(data=True) if d['layer_first_seen'] == current_layer - 1]

    for node in tqdm(nodes_current_layer, desc=f"Layer {current_layer}", unit="node"):
        references = refs_map.get(node, [])
        for ref in references:
            if ref not in G:
                direct_flag = 1 if current_layer == 1 else 0
                citation_count = get_total_citations(ref)
                G.add_node(ref, layer_first_seen=current_layer, freq=1, direct=direct_flag, 
                          citation_count=citation_count)
                new_nodes.append(ref)
            else:
                G.nodes[ref]['freq'] += 1
            G.add_edge(node, ref)

    # Update indegree
    for n in G.nodes:
        G.nodes[n]['indegree'] = G.in_degree(n)

    # ------------------------
    # Pruning rules
    # ------------------------
    to_remove = []
    for n in list(G.nodes):
        layer_seen = G.nodes[n]['layer_first_seen']
        freq = G.nodes[n]['freq']

        # Layer 2: remove if node does not appear in references of other layer 2 nodes
        if layer_seen == 2:
            appears_next = any(n in refs_map.get(parent, []) for parent in G.nodes if G.nodes[parent]['layer_first_seen'] == 2)
            if not appears_next:
                to_remove.append(n)
                continue

        # Layer 3+: remove if total citations <50 or low freq ratio
        if layer_seen >= 3:
            total_cites = get_total_citations(n)
            score = freq / total_cites if total_cites > 0 else 0
            if total_cites < 50 or score < 0.1:
                to_remove.append(n)

    for n in to_remove:
        if n in G:
            G.remove_node(n)
            if n in new_nodes:
                new_nodes.remove(n)

    return G, new_nodes

# ------------------------
# Dynamic GNN expansion & convergence
# ------------------------
def dynamic_gnn_expansion(source_doi, max_layers=5, top_k=3, epochs_per_layer=30):
    G = nx.DiGraph()
    citation_count = get_total_citations(source_doi)
    G.add_node(source_doi, layer_first_seen=0, freq=1, direct=1, indegree=0, citation_count=citation_count)

    max_layer_cap = max_layers
    converged = False
    current_layer = 1
    previous_top = []
    refs_cache = {}

    model = GraphSAGE(in_channels=5, hidden_channels=16, out_channels=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    while current_layer <= max_layer_cap and not converged:
        # Fetch references for current layer nodes
        current_layer_nodes = [n for n, d in G.nodes(data=True) if d['layer_first_seen'] == current_layer - 1]
        if not current_layer_nodes:
            break

        new_refs_map = get_references_from_doi_parallel(current_layer_nodes)
        refs_cache.update(new_refs_map)

        # Add layer with pruning
        G, new_nodes = add_layer_with_pruning(G, current_layer, refs_cache)
        if not new_nodes:
            if current_layer == 1:
                return None
            print(f"No new nodes added at layer {current_layer}, stopping expansion.")
            break

        # Convert to PyG data
        data, doi_list = pyg_data_from_nx(G)

        # Train GNN for this layer
        model.train()
        for epoch in range(epochs_per_layer):
            optimizer.zero_grad()
            embeddings = model(data.x, data.edge_index)
            if data.edge_index.size(1) > 0:
                loss = sum(F.mse_loss(embeddings[u], embeddings[v])
                           for u, v in zip(data.edge_index[0], data.edge_index[1]))
                loss.backward()
                optimizer.step()

        # Assign scores
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)
            scores = embeddings.norm(dim=1).cpu().numpy()
            for node, score in zip(doi_list, scores):
                G.nodes[node]['score'] = score

        # Check top-k convergence
        ranked = sorted(G.nodes(data=True), key=lambda x: x[1].get('score', 0), reverse=True)
        current_top = [node for node, _ in ranked[:top_k]]
        if previous_top and current_top == previous_top:
            print(f"Top-{top_k} converged at layer {current_layer}")
            converged = True
            break
        previous_top = current_top

        current_layer += 1

    return G, model

# ------------------------
# Test DOI
# ------------------------
test_doi = "10.1038/nature45678"
print(f"Building citation graph for DOI: {test_doi}")

G_test, model = dynamic_gnn_expansion(test_doi, max_layers=5, top_k=3)
if G_test:
    # Final scoring
    data_test, doi_list_test = pyg_data_from_nx(G_test)
    model.eval()
    with torch.no_grad():
        embeddings = model(data_test.x, data_test.edge_index)
        scores = embeddings.norm(dim=1).cpu().numpy()
        for node, score in zip(doi_list_test, scores):
            G_test.nodes[node]['score'] = score

    # Save CSV
    with open("citation_scores.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["DOI","Layer","Indegree","Direct","Frequency","Citation_Count","Score"])
        for node, data_node in G_test.nodes(data=True):
            writer.writerow([
                node, 
                data_node['layer_first_seen'], 
                data_node['indegree'],
                data_node['direct'], 
                data_node['freq'],
                data_node.get('citation_count', 0),
                round(data_node['score'], 4)
            ])
    print("Node scores saved to citation_scores.csv")

    # Layer weights
    layer_data = defaultdict(list)
    for node, data_node in G_test.nodes(data=True):
        layer_data[data_node['layer_first_seen']].append(data_node['score'])
    with open("layer_weights.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer","Num_Nodes","Sum_Score","Avg_Score"])
        for layer, scores in layer_data.items():
            writer.writerow([layer, len(scores), round(sum(scores),4), round(sum(scores)/len(scores),4)])
    print("Layer weights saved to layer_weights.csv")

    # Top 3 DOIs
    ranked = sorted(G_test.nodes(data=True), key=lambda x: x[1]['score'], reverse=True)
    print(f"\nTop 3 recommended articles for DOI {test_doi}:")
    for i, (node, data_node) in enumerate(ranked[:3]):
        print(f"{i+1}. DOI: {node} | Layer: {data_node['layer_first_seen']} | Score: {data_node['score']:.4f} | "
              f"Direct: {data_node['direct']} | Frequency: {data_node['freq']} | "
              f"Citations: {data_node.get('citation_count', 0)}")

    # Visualize
    if len(G_test.nodes) <= 50:  # Only visualize if not too large
        pos = nx.spring_layout(G_test, seed=42)
        node_sizes = [300 + G_test.nodes[n].get('score',0)*200 for n in G_test.nodes]
        node_colors = [G_test.nodes[n]['layer_first_seen'] for n in G_test.nodes]
        
        plt.figure(figsize=(12,7))
        nx.draw_networkx_nodes(G_test, pos, node_size=node_sizes, node_color=node_colors, 
                              cmap=plt.cm.viridis, alpha=0.8)
        nx.draw_networkx_edges(G_test, pos, edge_color='gray', alpha=0.5, arrows=True)
        
        # Label only the top 3 nodes
        top_nodes = [node for node, _ in ranked[:3]]
        labels = {node: f"{i+1}. {node[:15]}..." for i, node in enumerate(top_nodes)}
        nx.draw_networkx_labels(G_test, pos, labels, font_size=8)
        
        plt.title(f"Citation Graph Rooted at {test_doi} (Node size = GNN score)")
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Layer')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Graph too large ({len(G_test.nodes)} nodes) for visualization. See CSV file for results.")
else:
    print(f"Test DOI {test_doi} did not have sufficient citation network to expand.")