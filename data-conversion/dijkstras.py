# ------------------------
# Requirements:
# pip install networkx pandas
# ------------------------

import csv
import networkx as nx
import pandas as pd

# ------------------------
# Load scores and layers from CSV
# ------------------------
def load_scores_layers(score_csv):
    df = pd.read_csv(score_csv)
    scores = {}
    layers = {}
    for _, row in df.iterrows():
        doi = row['DOI'].strip()
        score = float(row['Score'])
        layer = int(row['Layer'])
        scores[doi] = score
        layers[doi] = layer
    return scores, layers

# ------------------------
# Load edges from CSV
# ------------------------
def load_edges(edges_csv):
    G = nx.DiGraph()
    with open(edges_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row.get("Citing_DOI")
            v = row.get("Cited_DOI")
            if u and v:
                G.add_edge(u, v)
    return G

# ------------------------
# Add nodes with scores and layers
# ------------------------
def add_nodes_to_graph(G, scores, layers):
    for doi, score in scores.items():
        layer = layers.get(doi, 0)
        G.add_node(doi, score=score, layer=layer)
    return G

# ------------------------
# Get top N scoring nodes (excluding source)
# ------------------------
def get_top_nodes(G, source_doi, top_n=10):
    nodes_data = [(n, d["score"]) for n, d in G.nodes(data=True) if n != source_doi]
    nodes_sorted = sorted(nodes_data, key=lambda x: x[1], reverse=True)
    return [n for n, _ in nodes_sorted[:top_n]]

# ------------------------
# Trace path back to source DOI
# ------------------------
def trace_path_to_source(G, node, source_doi):
    path = [node]
    current = node
    while current != source_doi:
        layer = G.nodes[current].get("layer", 0)
        if layer <= 0:
            break
        found = False
        for pred in G.predecessors(current):
            if G.nodes[pred].get("layer", 0) == layer - 1:
                path.append(pred)
                current = pred
                found = True
                break
        if not found:
            preds = list(G.predecessors(current))
            if preds:
                path.append(preds[0])
                current = preds[0]
            else:
                break
    path.reverse()
    return path

# ------------------------
# Save paths to CSV (only the arrays)
# ------------------------
def save_paths_only(G, source_doi, top_nodes, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Path_to_Source"])
        for node in top_nodes:
            path = trace_path_to_source(G, node, source_doi)
            # Convert to array string without quotes
            writer.writerow([f"[{', '.join(path)}]"])
    print(f"Paths saved to {output_csv}")

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    score_csv = "citation_scores_10.1038_nature14236.csv"  # DOI + Score + Layer CSV
    edges_csv = "citation_edges_10.1038_nature14236.csv"     # Edge CSV
    source_doi = "10.1038/nature14236"

    print("Loading scores and layers...")
    scores, layers = load_scores_layers(score_csv)

    print("Loading edges...")
    G = load_edges(edges_csv)

    print("Adding nodes with scores and layers...")
    G = add_nodes_to_graph(G, scores, layers)

    if source_doi not in G:
        print(f"[ERROR] Source DOI {source_doi} not found in graph")
        exit(1)

    print("Selecting top scoring nodes...")
    top_nodes = get_top_nodes(G, source_doi, top_n=10)

    print("Tracing paths back to source DOI...")
    save_paths_only(G, source_doi, top_nodes, "top10_paths_to_source_arrays.csv")
