"""

Load entities.csv + train/valid/test triples and produce:
 - id2idx.json, rel2id.json
 - data.pth (torch_geometric.data.Data) with x, edge_index, edge_type for training graph
 - split tensors for evaluation: train_triples.pt, valid_triples.pt, test_triples.pt

"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
from collections import OrderedDict
from typing import Tuple
try:
    from torch_geometric.data import Data
except Exception:
    Data = None

def read_entities(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    # Expect 'id' column
    if 'id' not in df.columns:
        raise ValueError("entities.csv must contain 'id' column")
    if 'name' not in df.columns:
        df['name'] = ""
    return df[['id','name'] + ([c for c in ['semantic_tag'] if c in df.columns])].copy()

def read_triples(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    cols_lower = {c.lower():c for c in df.columns}
    # Accept head/rel/tail or source/relation/target
    if 'head' in cols_lower and 'rel' in cols_lower and 'tail' in cols_lower:
        df = df.rename(columns={cols_lower['head']:'head', cols_lower['rel']:'rel', cols_lower['tail']:'tail'})
    elif 'source' in cols_lower and 'relation' in cols_lower and 'target' in cols_lower:
        df = df.rename(columns={cols_lower['source']:'head', cols_lower['relation']:'rel', cols_lower['target']:'tail'})
    else:
        raise ValueError(f"Triples file {path} must contain head/rel/tail or source/relation/target columns. Found: {list(df.columns)}")
    df['head'] = df['head'].astype(str).str.strip()
    df['tail'] = df['tail'].astype(str).str.strip()
    df['rel'] = df['rel'].astype(str).str.strip()
    return df[['head','rel','tail']].copy()

def build_mappings(entities_df: pd.DataFrame, triples_dfs: list) -> Tuple[OrderedDict, OrderedDict]:
    # id2idx based on entities file (preserves ordering)
    ids = list(entities_df['id'].astype(str).tolist())
    id2idx = OrderedDict((cid, i) for i, cid in enumerate(ids))
    # Collect relations across provided triples
    rels = OrderedDict()
    for df in triples_dfs:
        for r in df['rel'].unique().tolist():
            if r not in rels:
                rels[r] = len(rels)
    return id2idx, rels

def triples_to_idx(triples_df: pd.DataFrame, id2idx: dict, rel2id: dict, add_missing_nodes: bool=False):
    heads = []
    rels = []
    tails = []
    missing = set()
    for _, r in triples_df.iterrows():
        h = r['head']
        t = r['tail']
        rel = r['rel']
        if h not in id2idx or t not in id2idx:
            missing.add((h,t))
            if add_missing_nodes:
                # assign new idx for unseen nodes (rare; prefer to avoid)
                if h not in id2idx:
                    id2idx[h] = len(id2idx)
                if t not in id2idx:
                    id2idx[t] = len(id2idx)
        # After addition, if still missing raise error
        if h not in id2idx or t not in id2idx:
            raise KeyError(f"Node id missing in entities mapping: head={h} or tail={t}")
        heads.append(id2idx[h])
        tails.append(id2idx[t])
        if rel not in rel2id:
            # unseen relation: add to rel2id
            rel2id[rel] = len(rel2id)
        rels.append(rel2id[rel])
    return torch.tensor(heads, dtype=torch.long), torch.tensor(rels, dtype=torch.long), torch.tensor(tails, dtype=torch.long)

def load_or_create_feats(feats_path: str, n_nodes: int, feat_dim: int, out_save: str):
    if feats_path and os.path.exists(feats_path):
        feats = np.load(feats_path)
        if feats.shape[0] != n_nodes:
            # Try to handle if features are in different order or only partial
            print(f"[WARN] feats shape {feats.shape} doesn't match n_nodes={n_nodes}. Creating fresh random features.")
            feats = np.random.randn(n_nodes, feat_dim).astype(np.float32)
            #feats = model.encode(entity_names)
            np.save(out_save, feats)
            return feats
        return feats.astype(np.float32)
    else:
        print(f"[INFO] No node_feats provided. Generating random normal features (dim={feat_dim}) and saving to {out_save}")
        feats = np.random.randn(n_nodes, feat_dim).astype(np.float32)
        np.save(out_save, feats)
        return feats

def build_edge_index_from_triples(head_idx: torch.Tensor, tail_idx: torch.Tensor):
    # returns edge_index (2 x E) long tensor
    return torch.stack([head_idx, tail_idx], dim=0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entities", default= r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\entities.csv" , help="entities.csv path")
    p.add_argument("--train", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\train_val_test\train.csv", help="train triples CSV")
    p.add_argument("--valid", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\train_val_test\valid.csv", help="valid triples CSV")
    p.add_argument("--test", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\train_val_test\test.csv", help="test triples CSV")
    p.add_argument("--feats", default=None, help="Optional node_feats.npy path (N x D). If absent, random features created.")
    p.add_argument("--feat_dim", type=int, default=128, help="Feature dim to create if feats not provided")
    p.add_argument("--out_dir", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\pyg_dataset", help="output folder for dataset artifacts")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read data
    entities = read_entities(args.entities)
    train_df = read_triples(args.train)
    valid_df = read_triples(args.valid)
    test_df = read_triples(args.test)

    # Build mappings (ids from entities file)
    id2idx, rel2id = build_mappings(entities, [train_df, valid_df, test_df])

    # Save mappings
    id2idx_path = os.path.join(args.out_dir, "id2idx.json")
    rel2id_path = os.path.join(args.out_dir, "rel2id.json")
    with open(id2idx_path, "w", encoding="utf-8") as fh:
        json.dump(id2idx, fh, indent=2, ensure_ascii=False)
    with open(rel2id_path, "w", encoding="utf-8") as fh:
        json.dump(rel2id, fh, indent=2, ensure_ascii=False)
    print(f"[OK] Saved id2idx ({len(id2idx)}) -> {id2idx_path}")
    print(f"[OK] Saved rel2id ({len(rel2id)}) -> {rel2id_path}")

    # Convert triples to index form
    train_h, train_r, train_t = triples_to_idx(train_df, id2idx, rel2id, add_missing_nodes=False)
    valid_h, valid_r, valid_t = triples_to_idx(valid_df, id2idx, rel2id, add_missing_nodes=False)
    test_h, test_r, test_t = triples_to_idx(test_df, id2idx, rel2id, add_missing_nodes=False)

    # Build features
    n_nodes = len(id2idx)
    feats_save_path = os.path.join(args.out_dir, "node_feats.npy") if args.feats is None else args.feats
    feats = load_or_create_feats(args.feats, n_nodes, args.feat_dim, feats_save_path)
    # ensure order matches entities ordering (id2idx was built from entities ordering)
    # convert to torch tensor
    x = torch.tensor(feats, dtype=torch.float)

    # Build training graph edge_index and edge_type (only train edges)
    edge_index = build_edge_index_from_triples(train_h, train_t)  # shape [2, E_train]
    edge_type = train_r  # long tensor [E_train]

    # Wrap into PyG Data if available
    data_obj = None
    if Data is not None:
        data_obj = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        # add meta info
        data_obj.num_nodes = n_nodes

    # Save tensors for easy consumption by training script
    torch.save({'data_obj': data_obj, 'x': x, 'edge_index': edge_index, 'edge_type': edge_type},
               os.path.join(args.out_dir, "data.pth"))
    torch.save({'train': (train_h, train_r, train_t)},
               os.path.join(args.out_dir, "train_triples.pt"))
    torch.save({'valid': (valid_h, valid_r, valid_t)}, os.path.join(args.out_dir, "valid_triples.pt"))
    torch.save({'test': (test_h, test_r, test_t)}, os.path.join(args.out_dir, "test_triples.pt"))

    # Also write a small CSV summary of sizes
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"nodes: {n_nodes}\n")
        fh.write(f"train triples: {train_h.size(0)}\n")
        fh.write(f"valid triples: {valid_h.size(0)}\n")
        fh.write(f"test triples: {test_h.size(0)}\n")
        fh.write(f"feature dim: {x.shape[1]}\n")
        fh.write(f"relations: {len(rel2id)}\n")
    print("[OK] Saved data.pth, train/valid/test triples and summary in", args.out_dir)
    print("Done.")

if __name__ == "__main__":
    main()

