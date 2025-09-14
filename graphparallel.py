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

# ------------------------
# Crossref API helpers
# ------------------------
doi_cache = {}

def fetch_doi_data(doi):
    if doi in doi_cache:
        return doi_cache[doi]
    
    url = f"https://api.crossref.org/works/{doi}"
    headers = {'User-Agent': 'CitationGraph/1.0 (mailto:your-email@example.com)'}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()['message']
            doi_cache[doi] = data
            return data
    except Exception as e:
        print(f"Exception fetching {doi}: {str(e)}")
    doi_cache[doi] = None
    return None

def get_references(doi):
    data = fetch_doi_data(doi)
    if data and 'reference' in data:
        return [ref['DOI'].lower() for ref in data['reference'] if 'DOI' in ref and ref['DOI']]
    return []

def get_citation_count(doi):
    data = fetch_doi_data(doi)
    if data:
        return data.get('is-referenced-by-count', 1)
    return 1

# ------------------------
# Parallel fetching helpers
# ------------------------
def fetch_references_parallel(dois, max_workers=8):
    refs_map = {}
    def worker(doi):
        time.sleep(0.05)
        return doi, get_references(doi)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, doi) for doi in dois]
        for future in as_completed(futures):
            doi, refs = future.result()
            refs_map[doi] = refs
    return refs_map

def fetch_citation_counts_parallel(dois, max_workers=8):
    counts_map = {}
    def worker(doi):
        time.sleep(0.02)
        return doi, get_citation_count(doi)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, doi) for doi in dois]
        for future in as_completed(futures):
            doi, count = future.result()
            counts_map[doi] = count
    return counts_map

# ------------------------
# Graph expansion
# ------------------------
def add_layer_parallel(G, current_layer, max_refs_per_node=20):
    new_nodes = []
    layer_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == current_layer - 1]
    if not layer_nodes:
        return G, new_nodes

    references_dict = fetch_references_parallel(layer_nodes, max_workers=8)

    all_new_dois = set()
    for refs in references_dict.values():
        all_new_dois.update(refs)
    all_new_dois = [doi for doi in all_new_dois if doi not in G]
    citation_counts = fetch_citation_counts_parallel(all_new_dois, max_workers=8)

    for node in layer_nodes:
        references = references_dict.get(node, [])[:max_refs_per_node]
        for ref in references:
            if ref not in G:
                direct_flag = 1 if current_layer == 1 else 0
                citation_count = citation_counts.get(ref, 1)
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

    # Pruning
    current_layer_nodes = [n for n in G.nodes if G.nodes[n]['layer'] == current_layer]
    to_remove = []
    for n in current_layer_nodes:
        if G.nodes[n]['layer'] == 0 or G.nodes[n]['direct'] == 1:
            continue
        if current_layer >= 2:
            citing_papers = [u for u, v in G.in_edges(n) if G.nodes[u]['layer'] < current_layer]
            if len(citing_papers) < 2 and G.nodes[n]['citation_count'] < 10:
                to_remove.append(n)

    for n in to_remove:
        if n in G:
            G.remove_node(n)
            if n in new_nodes:
                new_nodes.remove(n)

    for n in G.nodes:
        G.nodes[n]['indegree'] = G.in_degree(n)

    return G, new_nodes

def build_initial_graph(source):
    G = nx.DiGraph()
    citation_count = get_citation_count(source)
    G.add_node(source, layer=0, direct=0, freq=1, seed=1, indegree=0, citation_count=citation_count)
    return G

# ------------------------
# Convert NetworkX graph to PyG data
# ------------------------
def pyg_data_from_nx(G):
    doi_list = list(G.nodes)
    node_features = []
    for node in doi_list:
        data = G.nodes[node]
        node_features.append([
            data['layer'], data['indegree'], data['direct'], 
            data['freq'], data['seed'], data.get('citation_count', 1)
        ])
    x = torch.tensor(node_features, dtype=torch.float)
    
    edge_index = []
    for u, v in G.edges:
        edge_index.append([doi_list.index(u), doi_list.index(v)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2,0), dtype=torch.long)
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
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# ------------------------
# Build citation graph
# ------------------------
def build_citation_graph(source_doi, max_layers=3, top_k=3, epochs_per_layer=50):
    source_data = fetch_doi_data(source_doi)
    if source_data is None:
        print(f"Source DOI {source_doi} not found")
        return None

    G = build_initial_graph(source_doi)
    current_layer = 1
    previous_top = []
    converged = False

    in_features = 6
    model = GraphSAGE(in_channels=in_features, hidden_channels=16, out_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    while current_layer <= max_layers and not converged:
        G, new_nodes = add_layer_parallel(G, current_layer)
        if not new_nodes:
            break

        data, doi_list = pyg_data_from_nx(G)

        # Train GNN
        model.train()
        for epoch in range(epochs_per_layer):
            optimizer.zero_grad()
            embeddings = model(data.x, data.edge_index)
            if data.edge_index.size(1) > 0:
                loss = sum(F.mse_loss(embeddings[u], embeddings[v]) for u, v in zip(data.edge_index[0], data.edge_index[1]))
                loss.backward()
                optimizer.step()

        # Assign scores
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)
            scores = embeddings.norm(dim=1).numpy() if data.edge_index.size(1) > 0 else [1.0]*len(doi_list)
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
# Run with real DOIs
# ------------------------
test_dois = [
    "10.1038/nature14236"
]

for source_doi in test_dois:
    print(f"\nBuilding citation graph for DOI: {source_doi}")
    G = build_citation_graph(source_doi, max_layers=3, top_k=3)
    if G is None:
        continue

    # Save node scores CSV
    with open(f"citation_scores_{source_doi.replace('/', '_')}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["DOI","Layer","Indegree","Direct","Freq","Seed","Citation_Count","Score"])
        for node, data_node in G.nodes(data=True):
            writer.writerow([
                node, data_node['layer'], data_node['indegree'], data_node['direct'],
                data_node['freq'], data_node['seed'], data_node.get('citation_count',0),
                round(data_node.get('score',0),4)
            ])
    print(f"Node scores saved.")

    # Layer weights CSV
    layer_data = defaultdict(list)
    for node, data_node in G.nodes(data=True):
        layer_data[data_node['layer']].append(data_node.get('score',0))
    with open(f"layer_weights_{source_doi.replace('/', '_')}.csv","w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer","Num_Nodes","Sum_Score","Avg_Score"])
        for layer, scores in layer_data.items():
            writer.writerow([layer, len(scores), round(sum(scores),4), round(sum(scores)/len(scores),4)])
    print(f"Layer weights saved.")

    # Top 3 recommended DOIs
    ranked = sorted(G.nodes(data=True), key=lambda x: x[1].get('score',0), reverse=True)
    print("\nTop 3 Recommended DOIs:")
    for i, (node, data_node) in enumerate(ranked[:10]):
        print(f"{i+1}. DOI: {node} | Layer: {data_node['layer']} | Score: {data_node.get('score',0):.4f} | "
              f"Indegree: {data_node['indegree']} | Direct: {data_node['direct']} | "
              f"Citations: {data_node.get('citation_count',0)}")

    # Visualize if small
    if len(G.nodes) <= 50:
        pos = nx.spring_layout(G, seed=42)
        node_sizes = [300 + G.nodes[n].get('score',0)*200 for n in G.nodes]
        node_colors = [G.nodes[n]['layer'] for n in G.nodes]
        plt.figure(figsize=(12,7))
        nx.draw_networkx_nodes(G,pos,node_size=node_sizes,node_color=node_colors,cmap=plt.cm.viridis,alpha=0.8)
        nx.draw_networkx_edges(G,pos,edge_color='gray',alpha=0.5,arrows=True)
        top_nodes = [node for node,_ in ranked[:3]]
        labels = {node: f"{i+1}. {node[:15]}..." for i,node in enumerate(top_nodes)}
        nx.draw_networkx_labels(G,pos,labels,font_size=8)
        plt.title(f"Citation Graph Rooted at {source_doi} (Node size = GNN score)")
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Layer')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"citation_graph_{source_doi.replace('/', '_')}.png")
        plt.show()
    else:
        print(f"Graph too large ({len(G.nodes)} nodes) for visualization.")
