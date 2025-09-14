# ------------------------
# Requirements:
# pip install torch torch-geometric networkx matplotlib
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

# ------------------------
# 1. Generate synthetic citation dataset with guaranteed edges
# ------------------------
num_total_dois = 20
references_big = {}

torch.manual_seed(42)

for i in range(num_total_dois):
    doi = f"DOI_{i}"
    if i == 0:
        # Ensure source DOI has outgoing references
        refs = ["DOI_1", "DOI_2", "DOI_3"]
    else:
        # Each DOI cites at least one previous DOI
        refs = [f"DOI_{j}" for j in range(i) if torch.rand(1).item() > 0.3]
        if len(refs) == 0:
            refs = [f"DOI_{i-1}"]
    references_big[doi] = refs

source_doi = "DOI_0"

# ------------------------
# 2. Graph expansion functions
# ------------------------
def add_layer(G, references, current_layer):
    new_nodes = []
    for node in list(G.nodes):
        if G.nodes[node]['layer'] == current_layer - 1:
            for ref in references.get(node, []):
                if ref not in G:
                    direct_flag = 1 if current_layer == 1 else 0
                    G.add_node(ref,
                               layer=current_layer,
                               direct=direct_flag,
                               freq=1,
                               seed=0)
                    new_nodes.append(ref)
                else:
                    G.nodes[ref]['freq'] += 1
                G.add_edge(node, ref)
    # update indegree
    for n in G.nodes:
        G.nodes[n]['indegree'] = G.in_degree(n)
    return G, new_nodes

def build_initial_graph(source):
    G = nx.DiGraph()
    G.add_node(source, layer=0, direct=0, freq=1, seed=1, indegree=0)
    return G

# ------------------------
# 3. Convert NetworkX graph to PyG data
# ------------------------
def pyg_data_from_nx(G):
    doi_list = list(G.nodes)
    node_features = []
    for node in doi_list:
        data = G.nodes[node]
        node_features.append([data['layer'], data['indegree'], data['direct'], data['freq'], data['seed']])
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
# 4. Define GraphSAGE GNN
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
# 5. Dynamic GNN expansion & convergence
# ------------------------
max_layer_cap = 10
top_k = 3
converged = False
current_layer = 1
previous_top = []

G = build_initial_graph(source_doi)
in_features = 5
model = GraphSAGE(in_channels=in_features, hidden_channels=16, out_channels=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs_per_layer = 100

while current_layer <= max_layer_cap and not converged:
    # Expand graph
    G, new_nodes = add_layer(G, references_big, current_layer)
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # If no edges, skip backward (nothing to train)

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
    ranked = sorted(G.nodes(data=True), key=lambda x: x[1]['score'], reverse=True)
    current_top = [node for node, _ in ranked[:top_k]]
    if previous_top and current_top == previous_top:
        print(f"Top-{top_k} converged at layer {current_layer}")
        converged = True
        break
    previous_top = current_top

    if not new_nodes:
        print(f"No new nodes added at layer {current_layer}, stopping expansion.")
        converged = True
        break

    current_layer += 1

# ------------------------
# 6. Save CSV files
# ------------------------
with open("citation_scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["DOI","Layer","Indegree","Direct","Freq","Seed","Score"])
    for node, data_node in G.nodes(data=True):
        writer.writerow([node,
                         data_node['layer'],
                         data_node['indegree'],
                         data_node['direct'],
                         data_node['freq'],
                         data_node['seed'],
                         round(data_node['score'],4)])
print("Node scores saved to citation_scores.csv")

layer_data = defaultdict(list)
for node, data_node in G.nodes(data=True):
    layer_data[data_node['layer']].append(data_node['score'])

with open("layer_weights.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Layer","Num_Nodes","Sum_Score","Avg_Score"])
    for layer, scores in layer_data.items():
        writer.writerow([layer, len(scores), round(sum(scores),4), round(sum(scores)/len(scores),4)])
print("Layer weights saved to layer_weights.csv")

# ------------------------
# 7. Print top 3 DOIs
# ------------------------
print("\nTop 3 Recommended DOIs:")
for node, data_node in ranked[:3]:
    print(f"DOI: {node} | Layer: {data_node['layer']} | Score: {data_node['score']:.4f} | Indegree: {data_node['indegree']} | Direct: {data_node['direct']}")

# ------------------------
# 8. Visualize graph
# ------------------------
pos = nx.spring_layout(G, seed=42)
node_sizes = [300 + G.nodes[n]['score']*200 for n in G.nodes]
node_colors = [G.nodes[n]['layer'] for n in G.nodes]

plt.figure(figsize=(12,7))
nx.draw_networkx(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, with_labels=True, arrows=True)
plt.title(f"Citation Graph Rooted at {source_doi} (Node size = GNN score)")
plt.show()
