# scripts/validate_kg.py
import pandas as pd
from pathlib import Path

conv = Path(r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data")
ents = pd.read_csv(conv/"entities.csv", dtype=str)
trip = pd.read_csv(conv/"triples.csv", dtype=str)

print("Entities:", len(ents))
print("Triples:", len(trip))

# duplicates
dups = ents['id'][ents['id'].duplicated()]
print("Duplicate entity ids:", dups.tolist())

# heads/tails present?
ids = set(ents['id'])
missing = set(trip['head']).union(trip['tail']) - ids
print("Node IDs present in triples but NOT in entities:", sorted(list(missing))[:10])

# relation distribution
print("Top relations:\n", trip['rel'].value_counts().head(20))
