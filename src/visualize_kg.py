#!/usr/bin/env python3
"""
Visualize a SNOMED-subgraph KG using NetworkX + matplotlib.

Inputs:
 - entities.csv (columns: id,name,semantic_tag (optional))
 - triples.csv OR triples.filtered.csv (columns: head,rel,tail)

Outputs:
 - out/kg.png          (visual PNG)
 - out/kg.gml          (graph file; open in Gephi)
 - out/legend.png      (optional legend image)

Example usage:
python src/visualize_kg.py --entities data/entities.csv --triples data/filtered/triples.filtered.csv --focus 233604007 --radius 2 --show_edge_labels --out_dir out

"""

import argparse
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import math

def read_entities(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    if not {"id","name"}.issubset(df.columns):
        raise ValueError("entities.csv must contain 'id' and 'name' columns")
    df["id"] = df["id"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    # optional semantic_tag
    if "semantic_tag" not in df.columns:
        df["semantic_tag"] = ""
    else:
        df["semantic_tag"] = df["semantic_tag"].astype(str).str.strip()
    return df.set_index("id").to_dict(orient="index")

def read_triples(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    # accept different column names
    cols = set(df.columns.str.lower())
    if "head" in cols and "rel" in cols and "tail" in cols:
        df = df.rename(columns={c:c.lower() for c in df.columns})
    elif {"source","relation","target"}.issubset(cols):
        df = df.rename(columns={"source":"head","relation":"rel","target":"tail"})
    elif {"source","rel","target"}.issubset(cols):
        df = df.rename(columns={"source":"head","rel":"rel","target":"tail"})
    else:
        # try common alt names
        raise ValueError("triples file must contain head/rel/tail or source/relation/target columns")
    df["head"] = df["head"].astype(str).str.strip()
    df["tail"] = df["tail"].astype(str).str.strip()
    df["rel"] = df["rel"].astype(str).str.strip()
    return df[["head","rel","tail"]].copy()

def build_graph(entities, triples):
    # Use MultiDiGraph to preserve possible parallel relations
    G = nx.MultiDiGraph()
    # add nodes with metadata
    for nid, meta in entities.items():
        G.add_node(nid, name=meta.get("name",""), semantic_tag=meta.get("semantic_tag",""))
    # add edges
    for _, row in triples.iterrows():
        h, r, t = row["head"], row["rel"], row["tail"]
        if not G.has_node(h):
            G.add_node(h, name="", semantic_tag="")
        if not G.has_node(t):
            G.add_node(t, name="", semantic_tag="")
        # store relation as edge attribute
        G.add_edge(h, t, relation=r)
    return G

def extract_focus_subgraph(G, focus_id=None, radius=1):
    if focus_id is None:
        return G
    if focus_id not in G:
        raise ValueError(f"Focus node {focus_id} not present in graph")
    # BFS up to radius (undirected expansion to capture parents/children)
    nodes = set([focus_id])
    frontier = {focus_id}
    for _ in range(radius):
        nbrs = set()
        for n in frontier:
            nbrs.update(set(G.predecessors(n)))
            nbrs.update(set(G.successors(n)))
        nbrs = nbrs - nodes
        if not nbrs:
            break
        nodes.update(nbrs)
        frontier = nbrs
    return G.subgraph(nodes).copy()

def color_map_for_tags(tags):
    # map each tag to a color
    unique = sorted(list(set(tags)))
    # matplotlib tab10 or tab20
    cmap = plt.get_cmap("tab20")
    mapping = {}
    for i, tag in enumerate(unique):
        mapping[tag] = cmap(i % 20)
    return mapping

def draw_graph(G, out_png, out_gml, show_edge_labels=False, label_nodes=True, figsize=(14,10), node_size=600):
    # Position using spring layout (suitable for small graphs). For hierarchical graphs, try shell_layout or kamada_kawai_layout.
    if len(G) == 0:
        raise ValueError("Graph is empty")

    # To make layout stable, use seed
    pos = nx.spring_layout(G, k=0.8/math.sqrt(max(1,len(G.nodes()))), seed=42)

    # Node attributes
    names = nx.get_node_attributes(G, "name")
    tags = nx.get_node_attributes(G, "semantic_tag")
    tag_list = [tags.get(n,"") for n in G.nodes()]

    tag2color = color_map_for_tags(tag_list)
    node_colors = [tag2color[tags.get(n,"")] for n in G.nodes()]

    plt.figure(figsize=figsize)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_size,
                           alpha=0.95)
    # Draw edges (directed arrows)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=15, connectionstyle='arc3,rad=0.05')

    # Node labels: use short names or IDs
    if label_nodes:
        labels = {}
        for n in G.nodes():
            nm = names.get(n,"")
            # shorten long names
            if nm == "" or nm is None:
                labels[n] = str(n)
            else:
                short = nm if len(nm) <= 20 else nm[:18]+".."
                labels[n] = short
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Edge labels: for MultiDiGraph we need to merge parallel edges into a single label
    if show_edge_labels:
        edge_labels = {}
        for u,v,data in G.edges(data=True):
            rel = data.get("relation","")
            # If multiple edges between u,v, combine
            key = (u,v)
            if key in edge_labels:
                if rel not in edge_labels[key]:
                    edge_labels[key] += f", {rel}"
            else:
                edge_labels[key] = rel
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[OK] Saved visualization PNG -> {out_png}")

    # write gml for opening in Gephi (note: MultiDiGraph -> convert to DiGraph by adding attrs)
    try:
        nx.write_gml(G, out_gml)
        print(f"[OK] Saved GML -> {out_gml}")
    except Exception as e:
        print("[WARN] Could not write GML file:", e)

    # Also return the mapping used for legend generation
    return tag2color

def save_legend(tag2color, out_path):
    # small legend image
    fig = plt.figure(figsize=(3, max(1, len(tag2color)*0.25)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 1.0
    for tag, color in tag2color.items():
        ax.add_patch(plt.Rectangle((0, y-0.15), width=0.2, height=0.12, color=color))
        ax.text(0.25, y-0.15, tag if tag!="" else "UNKNOWN", fontsize=9, va="bottom")
        y -= 0.18
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"[OK] Saved legend -> {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entities", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\entities.csv", help="entities.csv path")
    p.add_argument("--triples", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\filtered\triples.filtered.csv", help="triples.csv path (filtered preferred)")
    p.add_argument("--out_dir", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\out", help="output folder for images")
    p.add_argument("--focus", default=None, help="focus node id (e.g., 233604007) to plot neighborhood only")
    p.add_argument("--radius", type=int, default=1, help="radius for neighborhood around focus node")
    p.add_argument("--show_edge_labels", action="store_true", help="show relation names on edges (may overlap)")
    p.add_argument("--no_labels", action="store_true", help="do not show node labels (only colored dots)")
    p.add_argument("--figsize", type=float, nargs=2, default=(14,10), help="figure size")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    entities = read_entities(args.entities)
    triples = read_triples(args.triples)
    G = build_graph(entities, triples)

    if args.focus:
        G = extract_focus_subgraph(G, focus_id=str(args.focus), radius=args.radius)

    out_png = os.path.join(args.out_dir, "kg.png")
    out_gml = os.path.join(args.out_dir, "kg.gml")
    tag2color = draw_graph(G, out_png, out_gml, show_edge_labels=args.show_edge_labels, label_nodes=(not args.no_labels), figsize=tuple(args.figsize))
    # write legend
    legend_path = os.path.join(args.out_dir, "legend.png")
    save_legend(tag2color, legend_path)

if __name__ == "__main__":
    main()
