"""

Explain script for R-GCN trained with random 128-dim embeddings.

"""

import argparse
import os
import json
import torch
from torch import serialization
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque

# ----------------------------
# R-GCN Model Definition
# ----------------------------
try:
    from torch_geometric.nn import RGCNConv
except Exception:
    raise RuntimeError("torch_geometric is required. Install per official documentation.")


class RGCNLinkPredictor(nn.Module):
    def __init__(self, num_nodes, num_relations, in_dim, hidden_dim, num_layers=2):
        super().__init__()

        self.num_relations = num_relations
        self.emb = nn.Embedding(num_nodes, in_dim)

        self.layers = nn.ModuleList()
        self.layers.append(RGCNConv(in_dim, hidden_dim, num_relations))
        for _ in range(num_layers - 1):
            self.layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        self.rel_emb = nn.Embedding(num_relations, hidden_dim)

    def encode(self, x, edge_index, edge_type):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
            h = F.relu(h)
        return h

    def score_triplets(self, head_emb, rel_idx, tail_emb):
        r = self.rel_emb(rel_idx)
        return torch.sum(head_emb * r * tail_emb, dim=-1)


# ----------------------------
# Utility helpers
# ----------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_entities_map(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].astype(str)
    return dict(zip(df["id"], df["name"]))


def build_nx_graph(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df.rename(columns={df.columns[0]: "head", df.columns[1]: "rel", df.columns[2]: "tail"})
    G = nx.MultiDiGraph()
    for _, r in df.iterrows():
        G.add_edge(str(r["head"]), str(r["tail"]), relation=str(r["rel"]))
    return G


def compute_node_embeddings(model, device, x, edge_index, edge_type):
    with torch.no_grad():
        return model.encode(x.to(device), edge_index.to(device), edge_type.to(device))


def score_all_tails(head_idx, rel_idx, node_embs, model, device):
    with torch.no_grad():
        h = node_embs[head_idx].unsqueeze(0).to(device)
        r = model.rel_emb(torch.tensor([rel_idx], device=device))
        hr = (h * r).squeeze(0)
        scores = torch.matmul(node_embs.to(device), hr)
    return scores.cpu().numpy()


def aggregate_scores(head_idxs, rel_idx, node_embs, model, device, agg="sum"):
    arr = np.stack([score_all_tails(h, rel_idx, node_embs, model, device) for h in head_idxs], axis=0)
    if agg == "mean":
        return arr.mean(axis=0)
    if agg == "max":
        return arr.max(axis=0)
    return arr.sum(axis=0)


def find_paths(G, sources, target, max_hops=3, max_paths=20):
    paths = []
    for s in sources:
        q = deque([(s, [s])])
        count = 0
        while q and count < max_paths:
            node, path = q.popleft()
            if len(path) - 1 > max_hops:
                continue
            if node == target:
                paths.append(path)
                count += 1
                continue
            for nbr in G.successors(node):
                if nbr not in path:
                    q.append((nbr, path + [nbr]))
    return paths[:max_paths]


def score_path(path, G, node_embs, model, id2idx, rel2id, device):
    probs = []
    for u, v in zip(path[:-1], path[1:]):
        best = 0.0
        for _, d in G[u][v].items():
            rel = d["relation"]
            if rel not in rel2id:
                continue
            r_idx = rel2id[rel]
            with torch.no_grad():
                h = node_embs[id2idx[u]].unsqueeze(0).to(device)
                t = node_embs[id2idx[v]].unsqueeze(0).to(device)
                r = torch.tensor([r_idx], device=device)
                p = torch.sigmoid(model.score_triplets(h, r, t)).item()
                best = max(best, p)
        probs.append(best)
    score = 1.0
    for p in probs:
        score *= max(p, 1e-6)
    return score, probs


def path_to_text(path, G, id2name):
    parts = []
    for u, v in zip(path[:-1], path[1:]):
        rel = list(G[u][v].values())[0]["relation"]
        parts.append(f"{id2name.get(u,u)} --[{rel}]--> {id2name.get(v,v)}")
    return " ∧ ".join(parts)


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\pyg_dataset")
    p.add_argument("--checkpoint", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\src\checkpoints\rgcn\rgcn_epoch50.pt")
    p.add_argument("--entities", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\entities.csv")
    p.add_argument("--triples", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\filtered\triples.filtered.csv")
    p.add_argument("--heads", default="233604007")
    p.add_argument("--relation", default="Associated_Morphology")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--max_hops", type=int, default=3)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = args.device
    print(f"[INFO] device={device}")

    id2idx = load_json(os.path.join(args.dataset_dir, "id2idx.json"))
    rel2id = load_json(os.path.join(args.dataset_dir, "rel2id.json"))

    print("[INFO] loading dataset tensors...")
    '''pth_path = os.path.join(dataset_dir, "data_with_features.pth")
    # allowlist the PyG DataEdgeAttr class while loading
    with serialization.add_safe_globals([DataEdgeAttr]):
        data = torch.load(pth_path, map_location="cpu", weights_only=False)'''
    try:
        from torch_geometric.data.data import DataEdgeAttr
        from torch_geometric.data import Data as PyGData
        safe_globals = [DataEdgeAttr, PyGData]
    except Exception:
        # torch_geometric import failed? keep list empty (will still try weights_only=False)
        safe_globals = []

    data_pth = os.path.join(args.dataset_dir, "data.pth")

    # register allowlist *before* loading (works across PyTorch versions)
    if safe_globals:
        try:
            serialization.add_safe_globals(safe_globals)
        except Exception:
            # older/newer torch may return None; ignore — add_safe_globals still registers
            pass

    # Try loading with weights_only=False (needed for PyG objects)
    try:
        data = torch.load(data_pth, map_location="cpu", weights_only=False)
    except Exception as e:
        # If load still fails, print the error and attempt to show the pickle's top-level keys
        print("[WARN] torch.load failed with:", repr(e))
        print("[INFO] Attempting diagnostic load with weights_only=False to inspect keys (unsafe).")
        # Last-ditch attempt: load and inspect keys (only if you trust the file)
        data = torch.load(data_pth, map_location="cpu", weights_only=False)
        # fall through (if this still errors it will raise)
    # --- end loader ---

    '''data = torch.load(os.path.join(args.dataset_dir, "data.pth"), map_location="cpu")'''

    # Inspect loaded object and extract tensors reliably
    print("[INFO] Loaded object type:", type(data))
    if isinstance(data, dict):
        print("[INFO] Top-level keys:", list(data.keys()))
    # Common variants:
    # - data is a dict containing 'x','edge_index','edge_type'
    # - data is a dict: {'data_obj': Data(...), 'x':..., ...}
    # - data is a PyG Data object directly

    # Extract x, edge_index, edge_type robustly
    if isinstance(data, dict):
        if 'x' in data:
            x = data['x']
        elif 'data_obj' in data and hasattr(data['data_obj'], 'x'):
            x = data['data_obj'].x
        else:
            raise RuntimeError("Could not find 'x' in loaded data object; keys: " + ", ".join(list(data.keys())))
        if 'edge_index' in data:
            edge_index = data['edge_index']
        elif 'data_obj' in data and hasattr(data['data_obj'], 'edge_index'):
            edge_index = data['data_obj'].edge_index
        else:
            raise RuntimeError(
                "Could not find 'edge_index' in loaded data object; keys: " + ", ".join(list(data.keys())))
        if 'edge_type' in data:
            edge_type = data['edge_type']
        elif 'data_obj' in data and hasattr(data['data_obj'], 'edge_type'):
            edge_type = data['data_obj'].edge_type
        else:
            raise RuntimeError(
                "Could not find 'edge_type' in loaded data object; keys: " + ", ".join(list(data.keys())))
    elif hasattr(data, 'x') and hasattr(data, 'edge_index'):
        # data is a PyG Data object
        x = data.x
        edge_index = data.edge_index
        edge_type = getattr(data, 'edge_type', None)
        if edge_type is None:
            raise RuntimeError("Loaded PyG Data missing 'edge_type' attribute")
    else:
        raise RuntimeError(f"Unrecognized loaded data format: {type(data)}")

    # Ensure tensors
    x = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float)
    edge_index = edge_index if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index, dtype=torch.long)
    edge_type = edge_type if isinstance(edge_type, torch.Tensor) else torch.tensor(edge_type, dtype=torch.long)

    '''x = data["x"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]'''

    num_nodes = x.shape[0]
    num_relations = len(rel2id)

    print("[INFO] loading checkpoint:", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    '''hidden_dim = ckpt["rel_emb.weight"].shape[1]
    in_dim = ckpt["emb.weight"].shape[1]'''

    # Extract correct state_dict
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        raise KeyError("Checkpoint does not contain model_state_dict")

    # --- Infer dims from model_state_dict ---
    possible_rel_keys = [k for k in sd.keys() if "rel_emb" in k and "weight" in k]
    possible_emb_keys = [k for k in sd.keys() if "emb" in k and "weight" in k]

    if len(possible_rel_keys) == 0 or len(possible_emb_keys) == 0:
        print("[DEBUG] model_state_dict keys:", list(sd.keys())[:30])
        raise KeyError("Model state_dict does not contain rel_emb.weight or emb.weight keys")

    rel_key = possible_rel_keys[0]
    emb_key = possible_emb_keys[0]

    hidden_dim = sd[rel_key].shape[1]
    in_dim = sd[emb_key].shape[1]

    print(f"[INFO] Using rel_key={rel_key}")
    print(f"[INFO] Using emb_key={emb_key}")
    print(f"[INFO] checkpoint in_dim={in_dim}, hidden_dim={hidden_dim}")


    model = RGCNLinkPredictor(
        num_nodes=num_nodes,
        num_relations=num_relations,
        in_dim=in_dim,
        hidden_dim=hidden_dim
    )

    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()

    # *** THE IMPORTANT OPTION B FIX ***
    print("[INFO] Using model.emb.weight as node features (Option B).")
    x_use = model.emb.weight.to(device)

    print("[INFO] computing embeddings...")
    node_embs = compute_node_embeddings(model, device, x_use, edge_index, edge_type)

    id2name = load_entities_map(args.entities)
    G = build_nx_graph(args.triples)

    head_ids = [h.strip() for h in args.heads.split(",")]
    head_idxs = [id2idx[h] for h in head_ids]

    rel_idx = rel2id[args.relation]

    scores = aggregate_scores(head_idxs, rel_idx, node_embs, model, device)
    ranked = sorted([(i, s) for i, s in enumerate(scores)], key=lambda x: -x[1])

    print("\n=== Explanation Results ===")
    print("Heads:", head_ids)
    print("Relation:", args.relation)

    count = 0
    for idx, score in ranked:
        node_id = list(id2idx.keys())[list(id2idx.values()).index(idx)]
        if node_id in head_ids:
            continue

        paths = find_paths(G, head_ids, node_id, args.max_hops)
        scored = []
        for p in paths:
            pscore, probs = score_path(p, G, node_embs, model, id2idx, rel2id, device)
            scored.append((p, pscore, probs))

        scored.sort(key=lambda x: -x[1])
        best = scored[:3]

        print(f"\nRank #{count+1}: {id2name.get(node_id,node_id)} ({node_id}) score={score:.4f}")
        if not best:
            print("  No paths found.")
        for p, ps, pr in best:
            print(" ", path_to_text(p, G, id2name))
            print("    edge_probs:", pr)

        count += 1
        if count >= args.top_k:
            break


if __name__ == "__main__":
    main()
