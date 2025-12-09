"""

This script:
 - Load trained R-GCN checkpoint correctly
 - Uses the correct model_state_dict
 - Uses learned embeddings (Option B)
 - Performs BFS path search (3 hops)
 - Scores reachable nodes only
 - Prints clean explanations

"""

import json
import torch
from torch import serialization
from torch_geometric.data.data import DataEdgeAttr
from torch_geometric.nn import RGCNConv
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import networkx as nx
import numpy as np
import os
from collections import deque


# ---------------------------------------------------------
# R-GCN Model (MUST match training architecture)
# ---------------------------------------------------------
class RGCNLinkPredictor(nn.Module):
    def __init__(self, num_nodes, num_relations, in_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, in_dim)
        self.layers = nn.ModuleList([
            RGCNConv(in_dim, hidden_dim, num_relations)
        ])
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)

    def encode(self, x, edge_index, edge_type):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
            h = F.relu(h)
        return h

    def score_triplet(self, h_vec, rel_idx, t_vec):
        r = self.rel_emb(rel_idx)
        return torch.sum(h_vec * r * t_vec, dim=-1)


# ---------------------------------------------------------
# BFS path finder
# ---------------------------------------------------------
def find_paths(G, source, target, max_hops=3):
    paths = []
    q = deque([(source, [source])])

    while q:
        node, path = q.popleft()
        if len(path) - 1 > max_hops:
            continue
        if node == target and len(path) > 1:
            paths.append(path)
        for nxt in G.successors(node):
            if nxt not in path:
                q.append((nxt, path + [nxt]))

    return paths


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():

    print("\n=== Interactive KG Explanation ===")

    head_id = input("Enter HEAD concept ID (e.g., 233604007): ").strip()
    relation = input("Enter RELATION name (e.g., Associated_Morphology): ").strip()
    top_k = int(input("Enter top-k candidates (e.g., 5): "))

    dataset_dir = r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\pyg_dataset"
    triples_path = r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\filtered\triples.filtered.csv"
    entities_path = r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\entities.csv"
    checkpoint_path = r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\checkpoints\rgcn\rgcn_epoch50.pt"

    # Load mappings
    id2idx = json.load(open(os.path.join(dataset_dir, "id2idx.json")))
    rel2id = json.load(open(os.path.join(dataset_dir, "rel2id.json")))

    if head_id not in id2idx:
        print("[ERROR] Invalid head ID.")
        return
    if relation not in rel2id:
        print("[ERROR] Invalid relation name.")
        return

    # Load entity names
    df_ent = pd.read_csv(entities_path, dtype=str).fillna("")
    id2name = dict(zip(df_ent["id"], df_ent["name"]))

    # Build NetworkX graph
    df = pd.read_csv(triples_path, dtype=str)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(r["head"], r["tail"], relation=r["rel"])

    # Load dataset (PyG) safely
    serialization.add_safe_globals([DataEdgeAttr])

    data = torch.load(
        os.path.join(dataset_dir, "data.pth"),
        map_location="cpu",
        weights_only=False
    )

    # Extract tensors
    x = data["x"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]

    num_nodes = x.size(0)
    num_rel = len(rel2id)

    # ---------------------------------------------------------
    # LOAD CHECKPOINT — FIX FOR YOUR MODEL
    # ---------------------------------------------------------
    '''ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing model_state_dict")

    sd = ckpt["model_state_dict"]
    '''
    # ---------------- Robust checkpoint extraction ----------------
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = None

    if isinstance(ckpt, dict):
        # common container keys that might hold the real state_dict
        for cand in ("model_state_dict", "model", "state_dict", "model_weights", "model_state"):
            if cand in ckpt:
                sd = ckpt[cand]
                print(f"[INFO] Found weights under key: '{cand}'")
                break

        # if not found, maybe ckpt is already a raw state_dict (keys -> tensors)
        if sd is None:
            # test whether values are tensors (then it's probably a raw state_dict)
            sample_key = next(iter(ckpt.keys()))
            sample_val = ckpt[sample_key]
            if isinstance(sample_val, torch.Tensor):
                sd = ckpt
                print("[INFO] Checkpoint appears to be a raw state_dict (top-level tensors). Using it.")
            else:
                # nothing recognized — show keys for debugging
                print("[DEBUG] Checkpoint keys (first 60):", list(ckpt.keys())[:60])
                raise KeyError("Could not find model weights inside checkpoint. See printed keys above.")
    else:
        # unexpected checkpoint object type
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    # At this point 'sd' should be a mapping param_name -> tensor
    if not isinstance(sd, dict):
        raise RuntimeError("Extracted state_dict is not a dict. Aborting.")

    # Normalize keys: strip common prefixes like 'module.' or 'model.' if present
    def normalize_state_dict_keys(state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k
            # common wrappers added by DataParallel or saving patterns
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
            if new_k.startswith("model."):
                new_k = new_k[len("model."):]
            # some people save as 'net.' or 'backbone.' - remove small common wrappers conservatively
            if new_k.startswith("net."):
                new_k = new_k[len("net."):]
            new_sd[new_k] = v
        return new_sd

    sd = normalize_state_dict_keys(sd)

    # Quick sanity: try to find emb/rel keys inside sd
    found_emb = any("emb.weight" in k for k in sd.keys())
    found_rel = any("rel_emb.weight" in k for k in sd.keys())
    if not (found_emb and found_rel):
        print("[DEBUG] After normalization, state_dict keys (sample 60):", list(sd.keys())[:60])
        # don't abort yet — some models use different names. We'll attempt to infer dims safely later.
    else:
        print("[INFO] Found emb/rel keys in state_dict (good).")
    # ---------------- end robust extraction ----------------

    # infer dims
    hidden_dim = sd["rel_emb.weight"].shape[1]
    in_dim = sd["emb.weight"].shape[1]

    print(f"[INFO] Using in_dim={in_dim}, hidden_dim={hidden_dim}")

    model = RGCNLinkPredictor(num_nodes, num_rel, in_dim, hidden_dim)

    # load weights
    try:
        model.load_state_dict(sd, strict=True)
        print("[OK] Loaded model weights (strict=True)")
    except:
        print("[WARN] strict failed, trying strict=False")
        model.load_state_dict(sd, strict=False)

    model.eval()

    # ---------------------------------------------------------
    # OPTION B — USE TRAINED EMBEDDINGS (IGNORE x FEATURES)
    # ---------------------------------------------------------
    print("[INFO] Using model.emb.weight as node features")
    x_use = model.emb.weight

    with torch.no_grad():
        node_emb = model.encode(x_use, edge_index, edge_type)

    # ---------------------------------------------------------
    # SCORE ONLY REACHABLE NODES
    # ---------------------------------------------------------
    reachable = list(nx.descendants(G, head_id))
    print(f"[INFO] Reachable nodes: {len(reachable)}")

    head_idx = id2idx[head_id]
    rel_idx = rel2id[relation]
    r_tensor = torch.tensor([rel_idx])

    head_vec = node_emb[head_idx]

    candidates = []
    for nid in reachable:
        tail_idx = id2idx[nid]
        tail_vec = node_emb[tail_idx]
        score = model.score_triplet(head_vec, r_tensor, tail_vec).item()
        candidates.append((nid, score))

    candidates = sorted(candidates, key=lambda x: -x[1])[:top_k]

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    print("\n=== EXPLANATIONS ===")
    print(f"HEAD: {head_id} ({id2name.get(head_id, head_id)})")
    print(f"RELATION: {relation}")

    for rank, (cid, score) in enumerate(candidates, 1):
        cname = id2name.get(cid, cid)
        print(f"\nRank #{rank}: {cname} ({cid})  score={score:.4f}")

        paths = find_paths(G, head_id, cid, max_hops=3)
        if not paths:
            print("  No path within 3 hops.")
            continue

        for p in paths[:3]:
            ptext = " → ".join(id2name.get(n, n) for n in p)
            print("  Path:", ptext)


if __name__ == "__main__":
    main()
