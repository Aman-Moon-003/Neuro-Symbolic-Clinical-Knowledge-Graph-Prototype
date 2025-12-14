
"""
Create node feature matrix for entities.csv aligned with entity order.

Outputs:
 - node_feats.npy    (N x D)
 - node_feats_meta.json (info about dims and used models)

"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def compute_text_embeddings(names, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=32, device='cpu'):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Please install sentence-transformers: pip install -U sentence-transformers") from e
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(names, show_progress_bar=True, batch_size=batch_size)
    return np.array(embeddings, dtype=np.float32)

def build_graph_from_triples(triples_df, ids):
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(ids)
    for _, r in triples_df.iterrows():
        h, t = str(r['head']), str(r['tail'])
        if h not in G:
            G.add_node(h)
        if t not in G:
            G.add_node(t)
        G.add_edge(h, t)
    return G

def compute_node2vec_embeddings(G, dim=128, walk_length=20, num_walks=100, workers=4):
    try:
        from node2vec import Node2Vec
    except Exception as e:
        raise RuntimeError("Please install node2vec: pip install node2vec") from e
    node2vec = Node2Vec(G, dimensions=dim, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # return embeddings dict {node_id: vector}
    embs = {}
    for n in G.nodes():
        # ensure string key
        key = str(n)
        embs[key] = model.wv[key]
    return embs

def pca_reduce(X, k):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k)
    X_reduced = pca.fit_transform(X)
    return X_reduced.astype(np.float32), pca

def l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (X / norms).astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entities", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\entities.csv", help="entities.csv")
    p.add_argument("--triples", default=None, help="triples file")
    p.add_argument("--out_dir", default="data/converted", help="output folder")
    p.add_argument("--text_model", default="all-mpnet-base-v2", help="sentence-transformers model name (can prefix with 'sentence-transformers/')")
    p.add_argument("--batch_size", type=int, default=32, help="batch size for text embedding")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--use_node2vec", action="store_true", help="also compute node2vec features from triples")
    p.add_argument("--n2v_dim", type=int, default=128, help="node2vec dimension")
    p.add_argument("--n2v_walks", type=int, default=100, help="node2vec num_walks")
    p.add_argument("--n2v_walklen", type=int, default=20, help="node2vec walk_length")
    p.add_argument("--pca_dim", type=int, default=None, help="optional PCA final dimension (applied to concatenated features if set)")
    p.add_argument("--text_dim", type=int, default=None, help="force text embedding dimension if known (not necessary)")
    p.add_argument("--normalize", action="store_true", help="L2 normalize rows of final features")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entities_df = pd.read_csv(args.entities, dtype=str).fillna("")
    if 'id' not in entities_df.columns or 'name' not in entities_df.columns:
        raise ValueError("entities.csv must contain columns 'id' and 'name'")

    ids = entities_df['id'].astype(str).tolist()
    names = entities_df['name'].astype(str).tolist()

    print(f"[INFO] Found {len(ids)} entities. Building textual embeddings using model {args.text_model} on device {args.device} ...")
    # prepare model name for sentence-transformers (allow shortcut)
    model_name = args.text_model
    if not model_name.startswith("sentence-transformers/") and "/" not in model_name:
        model_name = f"sentence-transformers/{model_name}"

    text_embs = compute_text_embeddings(names, model_name=model_name, batch_size=args.batch_size, device=args.device)
    print(f"[OK] Text embeddings shape: {text_embs.shape}")

    features_list = [text_embs]
    feature_names = [f"text({model_name.split('/')[-1]})"]
    # optional node2vec
    if args.use_node2vec:
        if args.triples is None:
            raise ValueError("Node2Vec requested but --triples not provided.")
        print("[INFO] Building graph for node2vec from triples...")
        triples_df = pd.read_csv(args.triples, dtype=str).fillna("")
        G = build_graph_from_triples(triples_df, ids)
        print("[INFO] Computing node2vec embeddings (this may take a few minutes)...")
        n2v_dict = compute_node2vec_embeddings(G, dim=args.n2v_dim, walk_length=args.n2v_walklen, num_walks=args.n2v_walks)
        # create matrix aligned with entities order
        n2v_mat = np.zeros((len(ids), args.n2v_dim), dtype=np.float32)
        for i, nid in enumerate(ids):
            key = str(nid)
            if key in n2v_dict:
                n2v_mat[i] = n2v_dict[key]
            else:
                # fallback random vector
                n2v_mat[i] = np.random.randn(args.n2v_dim).astype(np.float32)
        features_list.append(n2v_mat)
        feature_names.append(f"node2vec({args.n2v_dim})")
        print(f"[OK] Node2Vec embeddings shape: {n2v_mat.shape}")

    # concatenate features
    X = np.concatenate(features_list, axis=1)
    print(f"[INFO] Concatenated feature shape: {X.shape}")

    # optional PCA to reduce to pca_dim
    pca_meta = None
    if args.pca_dim is not None:
        print(f"[INFO] Applying PCA to reduce to {args.pca_dim} dims...")
        X, pca_model = pca_reduce(X, args.pca_dim)
        pca_meta = {"n_components": args.pca_dim}
        print(f"[OK] PCA done. New shape: {X.shape}")

    # normalize rows
    if args.normalize:
        X = l2_normalize_rows(X)
        print("[OK] Row-wise L2 normalization applied.")

    # save
    node_feats_path = out_dir / "node_feats.npy"
    np.save(node_feats_path, X)
    meta = {
        "n_entities": len(ids),
        "feature_shape": X.shape,
        "text_model": model_name,
        "feature_parts": feature_names,
        "pca": pca_meta,
        "normalized": bool(args.normalize)
    }
    meta_path = out_dir / "node_feats_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[OK] Saved node_feats.npy -> {node_feats_path}")
    print(f"[OK] Saved metadata -> {meta_path}")
    print("Done.")

if __name__ == "__main__":
    main()

