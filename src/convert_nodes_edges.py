#!/usr/bin/env python3
"""

Convert nodes.csv + edges.csv into:

    - entities.csv   (entity_id, name)
    - triples.csv    (head, rel, tail)
    - concept_meta.json  (optional metadata dump)

This script is customized for the schema you uploaded:

nodes.csv columns:
    id,name,semantic_tag

edges.csv columns:
    source,relation,target

"""

import pandas as pd
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\edges.csv", help="Path to edges.csv")
    parser.add_argument("--out_dir", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data", help="Output Folder")
    parser.add_argument("--meta", action="store_true", help="If set, export concept_meta.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------------------------------------
    # 1. Load nodes.csv (schema already known)
    # -----------------------------------------------------------
    nodes = pd.read_csv(args.nodes, dtype=str)
    required_node_cols = {"id", "name", "semantic_tag"}

    if not required_node_cols.issubset(nodes.columns):
        raise ValueError(
            f"nodes.csv must contain columns: {required_node_cols}, "
            f"but found {set(nodes.columns)}"
        )

    # ensure clean strings
    nodes = nodes.fillna("").astype(str)
    nodes["id"] = nodes["id"].str.strip()
    nodes["name"] = nodes["name"].str.strip()

    # -----------------------------------------------------------
    # 2. Load edges.csv (schema also known)
    # -----------------------------------------------------------
    edges = pd.read_csv(args.edges, dtype=str)
    required_edge_cols = {"source", "relation", "target"}

    if not required_edge_cols.issubset(edges.columns):
        raise ValueError(
            f"edges.csv must contain columns: {required_edge_cols}, "
            f"but found {set(edges.columns)}"
        )

    edges = edges.fillna("").astype(str)
    edges["source"] = edges["source"].str.strip()
    edges["relation"] = edges["relation"].str.strip()
    edges["target"] = edges["target"].str.strip()

    # -----------------------------------------------------------
    # 3. Build entities.csv (id,name)
    # -----------------------------------------------------------
    entities = nodes[["id", "name"]].drop_duplicates().reset_index(drop=True)

    # If edges contain IDs not present in nodes, add them with empty names
    all_node_ids = set(entities["id"])
    edge_ids = set(edges["source"]).union(set(edges["target"]))
    missing_ids = sorted(edge_ids - all_node_ids)

    if missing_ids:
        print(f"[WARN] {len(missing_ids)} node IDs found in edges.csv but missing from nodes.csv.")
        print("       They will be added with empty names.")
        missing_df = pd.DataFrame({"id": missing_ids, "name": [""] * len(missing_ids)})
        entities = pd.concat([entities, missing_df], ignore_index=True)

    # -----------------------------------------------------------
    # 4. Build triples.csv (head,rel,tail)
    # -----------------------------------------------------------
    triples = edges.rename(columns={
        "source": "head",
        "relation": "rel",
        "target": "tail"
    })

    # -----------------------------------------------------------
    # 5. Save CSV outputs
    # -----------------------------------------------------------
    entities_path = os.path.join(args.out_dir, "entities.csv")
    triples_path = os.path.join(args.out_dir, "triples.csv")

    entities.to_csv(entities_path, index=False)
    triples.to_csv(triples_path, index=False)

    print(f"[OK] Wrote {entities_path} ({len(entities)} rows)")
    print(f"[OK] Wrote {triples_path} ({len(triples)} triples)")

    # -----------------------------------------------------------
    # 6. Optionally save metadata JSON
    # -----------------------------------------------------------
    if args.meta:
        meta = {}
        for _, row in nodes.iterrows():
            cid = row["id"]
            meta[cid] = {
                "name": row["name"],
                "semantic_tag": row["semantic_tag"]
            }

        meta_path = os.path.join(args.out_dir, "concept_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[OK] Wrote {meta_path}")

if __name__ == "__main__":
    main()
