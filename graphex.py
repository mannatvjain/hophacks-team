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

# ------------------------
# Crossref API helpers
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

def get_references(doi):
    data = fetch_doi_data(doi)
    if data and 'reference' in data:
        refs = []
        for ref in data['reference']:
            if 'DOI' in ref and ref['DOI']:
                refs.append(ref['DOI'].lower())
        return refs
    return []

def get_citation_count(doi):
    data = fetch_doi_data(doi)
    if data:
        return data.get('is-referenced-by-count', 1)
    return 1

# ------------------------
# 1. Graph expansion functions with real DOI data
# ------------------------
def add_layer(G, current_layer):
    new_nodes = []
    layer_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == current_layer - 1]
    
    for node in tqdm(layer_nodes, desc=f"Processing Layer {current_layer} nodes"):
        # Add polite delay to avoid rate limiting
        time.sleep(0.5)
        
        references = get_references(node)
        for ref in references:
            if ref not in G:
                direct_flag = 1 if current_layer == 1 else 0
                citation_count = get_citation_count(ref)
                G.add_node(ref,
                           layer=current_layer,
                           direct=direct_flag,
                           freq=1,
                           seed=0,
                           indegree=0,
                           citation_count=citation_count)
                new_nodes.append(ref)
            else:
                G.nodes[ref]['freq'] += 1
            G.add_edge(node, ref)
    
    # Apply pruning rules
    current_layer_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == current_layer]
    to_remove = []
    
    for n in current_layer_nodes:
        # Keep seed node and direct references
        if G.nodes[n]['layer'] == 0 or G.nodes[n]['direct'] == 1:
            continue
            
        # For layer 2+: remove if not cited enough or not connected to previous layers
        if current_layer >= 2:
            # Check if cited by at least 2 papers in the graph
            citing_papers = [u for u, v in G.in_edges(n) if G.nodes[u]['layer'] < current_layer]
            if len(citing_papers) < 2 and G.nodes[n]['citation_count'] < 10:
                to_remove.append(n)
    
    # Remove nodes that don't meet criteria
    for n in to_remove:
        if n in G:
            G.remove_node(n)
            if n in new_nodes:
                new_nodes.remove(n)
    
    # Update indegree for all nodes
    for n in G.nodes:
        G.nodes[n]['indegree'] = G.in_degree(n)
    
    return G, new_nodes

def build_initial_graph(source):
    G = nx.DiGraph()
    citation_count = get_citation_count(source)
    G.add_node(source, layer=0, direct=0, freq=1, seed=1, indegree=0, citation_count=citation_count)
    return G

# ------------------------
# 2. Convert NetworkX graph to PyG data
# ------------------------
def pyg_data_from_nx(G):
    doi_list = list(G.nodes)
    node_features = []
    for node in doi_list:
        data = G.nodes[node]
        node_features.append([
            data['layer'], 
            data['indegree'], 
            data['direct'], 
            data['freq'], 
            data['seed'],
            data.get('citation_count', 1)  # Add citation count as feature
        ])
    x = torch.tensor(node_features, dtype=torch.float)
    
    edge_index = []
    for u, v in G.edges:
        edge_index.append([doi_list.index(u), doi_list.index(v)])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index), doi_list

# ------------------------
# 3. Define GraphSAGE GNN
# ------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ------------------------
# 4. Dynamic GNN expansion & convergence
# ------------------------
def build_citation_graph(source_doi, max_layers=3, top_k=3, epochs_per_layer=100):
    # First check if source DOI exists
    source_data = fetch_doi_data(source_doi)
    if source_data is None:
        print(f"Source DOI {source_doi} not found in Crossref. Please use a valid DOI.")
        return None
    
    G = build_initial_graph(source_doi)
    
    max_layer_cap = max_layers
    converged = False
    current_layer = 1
    previous_top = []

    in_features = 6  # Updated to include citation_count
    model = GraphSAGE(in_channels=in_features, hidden_channels=16, out_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    while current_layer <= max_layer_cap and not converged:
        # Expand graph
        G, new_nodes = add_layer(G, current_layer)
        if not new_nodes:
            if current_layer == 1:
                print(f"No references found for source DOI {source_doi}")
                return None
            print(f"No new nodes added at layer {current_layer}, stopping expansion.")
            break

        # Convert to PyG data
        data, doi_list = pyg_data_from_nx(G)

        # Train GNN
        model.train()
        for epoch in range(epochs_per_layer):
            optimizer.zero_grad()
            embeddings = model(data.x, data.edge_index)
            if data.edge_index.size(1) > 0:
                loss = None
                for u, v in zip(data.edge_index[0], data.edge_index[1]):
                    edge_loss = F.mse_loss(embeddings[u], embeddings[v])
                    if loss is None:
                        loss = edge_loss
                    else:
                        loss = loss + edge_loss
                if loss is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # Assign scores
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)
            if data.edge_index.size(1) == 0:
                for node in doi_list:
                    G.nodes[node]['score'] = 1.0
            else:
                scores = embeddings.norm(dim=1).numpy()
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

    return G

# ------------------------
# 5. Run with a real DOI
# ------------------------
# Try these DOIs which are known to have references
test_dois = [
    "10.1038/nature14236",        # A highly cited Nature paper
    "10.1126/science.1257565",    # A highly cited Science paper
    "10.1016/j.cell.2016.11.038", # A highly cited Cell paper
]

for source_doi in test_dois:
    print(f"\nBuilding citation graph for DOI: {source_doi}")
    
    G = build_citation_graph(source_doi, max_layers=3, top_k=3)
    if G is None:
        print(f"Skipping DOI {source_doi} due to errors")
        continue

    # ------------------------
    # 6. Save CSV files
    # ------------------------
    with open(f"citation_scores_{source_doi.replace('/', '_')}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["DOI","Layer","Indegree","Direct","Freq","Seed","Citation_Count","Score"])
        for node, data_node in G.nodes(data=True):
            writer.writerow([
                node,
                data_node['layer'],
                data_node['indegree'],
                data_node['direct'],
                data_node['freq'],
                data_node['seed'],
                data_node.get('citation_count', 0),
                round(data_node.get('score', 0), 4)
            ])
    print(f"Node scores saved to citation_scores_{source_doi.replace('/', '_')}.csv")

    # Layer weights
    layer_data = defaultdict(list)
    for node, data_node in G.nodes(data=True):
        layer_data[data_node['layer']].append(data_node.get('score', 0))
    with open(f"layer_weights_{source_doi.replace('/', '_')}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer","Num_Nodes","Sum_Score","Avg_Score"])
        for layer, scores in layer_data.items():
            writer.writerow([layer, len(scores), round(sum(scores),4), round(sum(scores)/len(scores),4)])
    print(f"Layer weights saved to layer_weights_{source_doi.replace('/', '_')}.csv")

    # ------------------------
    # 7. Print top 3 DOIs
    # ------------------------
    ranked = sorted(G.nodes(data=True), key=lambda x: x[1].get('score', 0), reverse=True)
    print(f"\nTop 3 Recommended DOIs for {source_doi}:")
    for i, (node, data_node) in enumerate(ranked[:3]):
        print(f"{i+1}. DOI: {node} | Layer: {data_node['layer']} | Score: {data_node.get('score', 0):.4f} | "
              f"Indegree: {data_node['indegree']} | Direct: {data_node['direct']} | "
              f"Citations: {data_node.get('citation_count', 0)}")

    # ------------------------
    # 8. Visualize graph
    # ------------------------
    if len(G.nodes) <= 50:  # Only visualize if not too large
        pos = nx.spring_layout(G, seed=42)
        node_sizes = [300 + G.nodes[n].get('score', 0)*200 for n in G.nodes]
        node_colors = [G.nodes[n]['layer'] for n in G.nodes]

        plt.figure(figsize=(12,7))
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                              cmap=plt.cm.viridis, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True)
        
        # Label only the top 3 nodes
        top_nodes = [node for node, _ in ranked[:3]]
        labels = {node: f"{i+1}. {node[:15]}..." for i, node in enumerate(top_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Citation Graph Rooted at {source_doi} (Node size = GNN score)")
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Layer')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"citation_graph_{source_doi.replace('/', '_')}.png")
        plt.show()
    else:
        print(f"Graph too large ({len(G.nodes)} nodes) for visualization. See CSV file for results.")