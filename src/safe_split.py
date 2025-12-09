"""

Create safe train/val/test splits for KG triples preserving:
 - minimum training examples per relation
 - no node leakage: every node in val/test appears in training (unless allow_inductive_test=True)
 - training graph weakly connected (attempt to ensure)

Usage:
    python src/safe_split.py
      --triples data/filtered/triples.filtered.csv
      --out_dir data/splits
      --min_train_per_rel 3
      --val_frac 0.1
      --test_frac 0.1
      --seed 42

Outputs (written to out_dir):
    train.csv (head,rel,tail)
    valid.csv
    test.csv
    summary.txt (human-readable)

"""

import argparse
import os
import pandas as pd
import random
from collections import defaultdict, Counter
import networkx as nx

def read_triples(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    # standardize columns
    cols = {c.lower(): c for c in df.columns}
    # try to detect columns
    if "head" in cols and "rel" in cols and "tail" in cols:
        df = df.rename(columns={cols['head']:'head', cols['rel']:'rel', cols['tail']:'tail'})
    elif "source" in cols and "relation" in cols and "target" in cols:
        df = df.rename(columns={cols['source']:'head', cols['relation']:'rel', cols['target']:'tail'})
    else:
        raise ValueError("triples file must contain head/rel/tail or source/relation/target")
    df['head'] = df['head'].astype(str).str.strip()
    df['tail'] = df['tail'].astype(str).str.strip()
    df['rel'] = df['rel'].astype(str).str.strip()
    return df[['head','rel','tail']].copy()

def ensure_node_coverage(train_set, nodes_needed, df):
    """
    Helper: Move edges from df (a DataFrame slice) into train_set if they involve nodes in nodes_needed.
    Returns the updated df (rows not moved) and updated train_set list.
    """
    remaining_rows = []
    moved = []
    for _, r in df.iterrows():
        if r['head'] in nodes_needed or r['tail'] in nodes_needed:
            train_set.append((r['head'], r['rel'], r['tail']))
            moved.append((r['head'], r['rel'], r['tail']))
        else:
            remaining_rows.append((r['head'], r['rel'], r['tail']))
    return remaining_rows, moved

def write_csv(rows, path):
    df = pd.DataFrame(rows, columns=['head','rel','tail'])
    df.to_csv(path, index=False)

def build_weak_graph(triples_rows):
    G = nx.DiGraph()
    for h,r,t in triples_rows:
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h, t)
    return G

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triples", default= r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\filtered\triples.filtered.csv", help="Path to triples.csv")
    p.add_argument("--out_dir", default= r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\train_val_test", help="Output folder")
    p.add_argument("--min_train_per_rel", type=int, default=3, help="Minimum edges per relation to reserve into train")
    p.add_argument("--val_frac", type=float, default=0.1, help="Validation fraction (of total triples)")
    p.add_argument("--test_frac", type=float, default=0.1, help="Test fraction (of total triples)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--allow_inductive_test", action="store_true", help="If set, allow val/test nodes that do NOT appear in train (inductive). Default: False")
    args = p.parse_args()

    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    df = read_triples(args.triples)
    n_total = len(df)
    print(f"[INFO] Loaded {n_total} triples from {args.triples}")

    # relation groups
    rel2rows = defaultdict(list)
    for _, r in df.iterrows():
        rel2rows[r['rel']].append((r['head'], r['rel'], r['tail']))

    # 1) Reserve minimum per relation into train
    train = []
    pool = []  # remaining edges eligible for splitting
    for rel, rows in rel2rows.items():
        if len(rows) <= args.min_train_per_rel:
            # put all into train
            train.extend(rows)
        else:
            # shuffle and reserve min_train_per_rel into train
            random.shuffle(rows)
            train.extend(rows[:args.min_train_per_rel])
            pool.extend(rows[args.min_train_per_rel:])

    # 2) Shuffle remaining pool and assign provisional splits according to fractions
    random.shuffle(pool)
    n_val_target = int(args.val_frac * n_total)
    n_test_target = int(args.test_frac * n_total)
    # We'll try to fill val/test up to targets but we will avoid placing edges that create unseen nodes in val/test
    val = []
    test = []
    # current nodes in train
    train_nodes = set([h for h,_,_ in train] + [t for _,_,t in train])

    for (h,rel,t) in pool:
        # if test target not full, candidate for test
        assigned = False
        # function to check if both nodes exist in train already (to avoid node leakage)
        def nodes_in_train(h,t):
            return (h in train_nodes) and (t in train_nodes)
        # Prefer assigning to test, then val, else train
        if len(test) < n_test_target:
            if args.allow_inductive_test or nodes_in_train(h,t):
                test.append((h,rel,t)); assigned=True
        if not assigned and len(val) < n_val_target:
            if args.allow_inductive_test or nodes_in_train(h,t):
                val.append((h,rel,t)); assigned=True
        if not assigned:
            # can't safely put into val/test (would create unseen nodes) -> put into train
            train.append((h,rel,t))
            train_nodes.add(h); train_nodes.add(t)

    # After first pass, if val/test are below targets because of safety, it's fine; we prefer safety.
    print(f"[INFO] After first pass: train={len(train)}, val={len(val)}, test={len(test)} (targets val={n_val_target}, test={n_test_target})")

    # 3) Ensure training graph is weakly connected (single weak component)
    # If not connected, try to move some edges from val/test into train to connect components.
    train_graph = build_weak_graph(train)
    n_comp = nx.number_weakly_connected_components(train_graph) if len(train_graph) > 0 else 0
    if n_comp > 1:
        print(f"[WARN] Training graph has {n_comp} weak components. Attempting to connect components by moving edges from val/test -> train.")
        # compute components and get their node sets
        comps = list(nx.weakly_connected_components(train_graph))
        comp_node_to_idx = {}
        for i, comp in enumerate(comps):
            for n in comp:
                comp_node_to_idx[n] = i
        # function to try move an edge that connects different components
        def try_move_to_connect(candidate_list):
            moved = False
            for idx, (h,rel,t) in enumerate(list(candidate_list)):
                hi = comp_node_to_idx.get(h, None)
                ti = comp_node_to_idx.get(t, None)
                # if either endpoint is in train but in different components OR one endpoint in train and other not in train, this edge may connect
                if hi is not None and ti is not None and hi != ti:
                    # move this edge to train
                    train.append((h,rel,t))
                    # remove from candidate_list
                    candidate_list.pop(idx)
                    # update structures
                    train_graph.add_node(h); train_graph.add_node(t); train_graph.add_edge(h,t)
                    # recompute comps quickly (could be expensive but small graphs ok)
                    new_comps = list(nx.weakly_connected_components(train_graph))
                    # rebuild mapping
                    comp_node_to_idx.clear()
                    for i, comp in enumerate(new_comps):
                        for n in comp:
                            comp_node_to_idx[n] = i
                    moved = True
                    break
            return moved

        # attempt moves from val first, then test, up to a reasonable limit
        changed = True
        attempts = 0
        while nx.number_weakly_connected_components(train_graph) > 1 and attempts < 1000 and changed:
            changed = False
            if try_move_to_connect(val):
                changed = True
            elif try_move_to_connect(test):
                changed = True
            attempts += 1
        n_comp_after = nx.number_weakly_connected_components(train_graph)
        print(f"[INFO] After attempts, training components: {n_comp_after}")

    # 4) Final safety checks: ensure no node in val/test is absent from train (unless allow_inductive_test)
    train_nodes = set([h for h,_,_ in train] + [t for _,_,t in train])
    def check_and_fix(candidate_list, list_name):
        moved_count = 0
        new_list = []
        for (h,rel,t) in list(candidate_list):
            if args.allow_inductive_test:
                new_list.append((h,rel,t))
            else:
                if (h in train_nodes) and (t in train_nodes):
                    new_list.append((h,rel,t))
                else:
                    # move to train
                    train.append((h,rel,t))
                    train_nodes.add(h); train_nodes.add(t)
                    moved_count += 1
        return new_list, moved_count

    val, moved_val = check_and_fix(val, "val")
    test, moved_test = check_and_fix(test, "test")
    if moved_val + moved_test > 0:
        print(f"[INFO] Moved {moved_val} val + {moved_test} test edges into train to preserve node coverage.")

    # 5) Final reporting and writing
    n_train = len(train); n_val = len(val); n_test = len(test)
    print(f"[INFO] Final counts: train={n_train}, val={n_val}, test={n_test} (total={n_train+n_val+n_test})")
    # compute relation distributions
    def rel_counts(rows):
        c = Counter([r for (_,r,_) in rows])
        return c

    out_train = os.path.join(args.out_dir, "train.csv")
    out_val = os.path.join(args.out_dir, "valid.csv")
    out_test = os.path.join(args.out_dir, "test.csv")
    write_csv(train, out_train)
    write_csv(val, out_val)
    write_csv(test, out_test)

    # write summary
    summary = []
    summary.append(f"Input triples: {n_total}\n")
    summary.append(f"Train: {n_train}  Val: {n_val}  Test: {n_test}\n")
    summary.append("\nRelation counts (train):\n")
    for r,c in rel_counts(train).most_common():
        summary.append(f"{r:30s} : {c}\n")
    summary.append("\nRelation counts (val):\n")
    for r,c in rel_counts(val).most_common():
        summary.append(f"{r:30s} : {c}\n")
    summary.append("\nRelation counts (test):\n")
    for r,c in rel_counts(test).most_common():
        summary.append(f"{r:30s} : {c}\n")
    # connectedness
    train_graph = build_weak_graph(train)
    summary.append(f"\nTraining graph weak components: {nx.number_weakly_connected_components(train_graph)}\n")
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.writelines(summary)
    print(f"[OK] Wrote splits to {args.out_dir} and summary to {summary_path}")

if __name__ == "__main__":
    main()
