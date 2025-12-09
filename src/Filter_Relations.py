
"""

Filter / merge / repurpose rare relations in triples.csv

# 1) Drop relations that appear fewer than min_count (e.g. 3)
python src/filter_relations.py --triples data/triples.csv --out_dir data/filtered --mode drop --min_count 3

# 2) Merge rare relations into a generic label "OTHER"
python src/filter_relations.py --triples data/triples.csv --out_dir data/filtered --mode merge --min_count 3 --merge_label OTHER

# 3) Repurpose rare relations into a separate file (kept for explanation only)
python src/filter_relations.py --triples data/triples.csv --out_dir data/filtered --mode repurpose --min_count 3

Notes:
 - The script preserves order of rows.
 - It writes:
    * filtered/triples.csv       -> triples for training (depending on mode)
    * filtered/rare_triples.csv  -> triples considered rare (if repurpose or merge)
    * filtered/report.txt        -> human-readable summary
    
"""

import argparse
import os
import pandas as pd
import re
from collections import Counter, OrderedDict

def normalize_relation(r: str) -> str:
    """Normalize relation string: trim, unify spaces/underscores, lowercase common patterns, fix simple typos."""
    if pd.isna(r):
        return ""
    s = str(r).strip()
    # replace spaces and multiple non-alnum with single underscore
    s = re.sub(r'[^0-9A-Za-z]+', '_', s)
    s = re.sub(r'__+', '_', s)
    # correct common accidental typos (example from your data)
    s = s.replace("Fiding_Site", "Finding_Site")
    # normalize case: keep original-style but unify variants by title-case underscores
    # We'll use uppercase words separated by single underscore for readability
    parts = [p.capitalize() for p in s.split('_') if p != ""]
    if len(parts) == 0:
        return ""
    return "_".join(parts)

#path = r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data"

def load_triples(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    # require columns head,rel,tail (or rel alternatives)
    rel_col = None
    for candidate in ["rel","relation","predicate","Relation","REL"]:
        if candidate in df.columns:
            rel_col = candidate
            break
    if rel_col is None:
        raise ValueError("Could not find relation column in triples file. Expected 'rel' or 'relation'.")
    # standardize columns to head, rel, tail
    if 'head' not in df.columns or 'tail' not in df.columns:
        # try alternatives
        if 'source' in df.columns and 'target' in df.columns:
            df = df.rename(columns={'source':'head','target':'tail'})
        else:
            # leave as-is and assume user used head/tail
            pass
    df = df.rename(columns={rel_col:'rel'})
    return df[['head','rel','tail']].copy()

def write_report(out_dir, before_counts, after_counts, rare_rels, mode, min_count, merge_label):
    rpt = []
    rpt.append("Relation filtering report\n")
    rpt.append(f"Mode: {mode}\n")
    rpt.append(f"min_count: {min_count}\n")
    if mode == "merge":
        rpt.append(f"merge_label: {merge_label}\n")
    rpt.append("\nTop relations BEFORE filtering:\n")
    for r,c in before_counts.most_common(30):
        rpt.append(f"{r:35s} : {c}\n")
    rpt.append("\nRelations marked as RARE (below min_count):\n")
    for r,c in rare_rels:
        rpt.append(f"{r:35s} : {c}\n")
    rpt.append("\nTop relations AFTER filtering/merging:\n")
    for r,c in after_counts.most_common(30):
        rpt.append(f"{r:35s} : {c}\n")
    rpt_path = os.path.join(out_dir, "report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.writelines(rpt)
    print("[OK] Wrote report:", rpt_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triples", required=True, help="Path to triples.csv (head,rel,tail)")
    p.add_argument("--out_dir", required=True, help="Output folder")
    p.add_argument("--min_count", type=int, default=3, help="Minimum frequency for a relation to be considered 'frequent'")
    p.add_argument("--mode", choices=["drop","merge","repurpose"], default="repurpose",
                   help="What to do with rare relations: drop them, merge them to a generic rel, or repurpose into a separate file")
    p.add_argument("--merge_label", default="OTHER", help="If mode=merge, label to use for merged rare relations")
    p.add_argument("--normalize", action="store_true", help="Normalize relation names (case/spacing fixes)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_triples(args.triples)
    print(f"[INFO] Loaded triples: {len(df)} rows")

    # Normalize relation values if requested
    if args.normalize:
        df['rel_orig'] = df['rel']
        df['rel'] = df['rel'].apply(normalize_relation)
    else:
        # still strip whitespace
        df['rel'] = df['rel'].astype(str).str.strip()

    # Frequency counts
    before_counts = Counter(df['rel'].tolist())
    # Determine rare relations
    rare_rels = sorted([(r,c) for r,c in before_counts.items() if c < args.min_count], key=lambda x: x[1])
    rare_rel_set = set(r for r,_ in rare_rels)
    print(f"[INFO] Found {len(before_counts)} distinct relations; {len(rare_rels)} are below min_count={args.min_count}")

    # Decide what to do per mode
    if args.mode == "drop":
        kept_df = df[~df['rel'].isin(rare_rel_set)].copy()
        rare_df = df[df['rel'].isin(rare_rel_set)].copy()
    elif args.mode == "merge":
        # Replace rare relations with merge_label
        df2 = df.copy()
        df2.loc[df2['rel'].isin(rare_rel_set), 'rel'] = args.merge_label
        kept_df = df2.copy()
        rare_df = df[df['rel'].isin(rare_rel_set)].copy()
    else:  # repurpose
        kept_df = df[~df['rel'].isin(rare_rel_set)].copy()
        rare_df = df[df['rel'].isin(rare_rel_set)].copy()

    after_counts = Counter(kept_df['rel'].tolist())

    # Write outputs
    out_triples = os.path.join(args.out_dir, "triples.filtered.csv")
    kept_df[['head','rel','tail']].to_csv(out_triples, index=False)
    print("[OK] Wrote filtered triples for training:", out_triples, " (rows:", len(kept_df), ")")

    rare_path = os.path.join(args.out_dir, "triples.rare.csv")
    rare_df[['head','rel','tail']].to_csv(rare_path, index=False)
    print("[OK] Wrote rare triples (structural/explaining):", rare_path, " (rows:", len(rare_df), ")")

    write_report(args.out_dir, before_counts, after_counts, rare_rels, args.mode, args.min_count, args.merge_label)
    print("[DONE] Filtering complete.")

if __name__ == "__main__":
    main()
