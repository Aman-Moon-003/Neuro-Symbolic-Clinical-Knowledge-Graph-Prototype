# Neuro-Symbolic-Clinical-Knowledge-Graph-Prototype
This repository is academically oriented prototype for MTech Thesis at IIT KGP.

It contains following features:

● A small hand curated cinically-aligned knowledge graph build by using SNOMED-CT browser.

● Integrated the Relational Graph Convolutional Network (R-GCN).

● Fuzzy clinical rules for neuro-symbolic reasoning.

● Minimal reproducible workflow with no UI components.

Once SNOMED-CT RF2 access is approved, the current manually curated graph will be replaced by an RF2-derived KG.
This repo intentionally focuses on clarity, reproducibility, and scientific structure rather than UI/production engineering.

**Research Goals:**

This prototype supports the thesis:

**“Neuro-Symbolic AI for Clinical Knowledge Graph Reasoning.”**

The objectives are:

● To build a clinically meaningful KG using SNOMED-CT concepts.

● To train R-GCN or similar GNN models.

● To integrate fuzzy logic rules using Łukasiewicz t-norms.

● To provide simple and human readable reasoning explanations.

**SNOMED CT Licensing Notes:**

This repository does not contain:

● SNOMED CT RF2 files.

● Raw SNOMED distributions.

● Redistributable SNOMED components.

It only has hand-curated labels and synthetic edges for prototyping.

See 'docs/license_notes.md' for details.

**Repository Structure:**

```bash
├── data/
│   ├── seeds.csv            # selected respiratory concepts (labels or SCTIDs)
│   ├── entities.csv         # prototype node list
│   ├── triples.csv          # prototype edge list
│   └── node_feats.npy       # BERT-derived node embeddings (or created via script)
│
├── notebooks/
│   ├── 01_build_kg.ipynb    # NetworkX KG build + visualization
│   └── 03_rgcn_train_eval.ipynb  # R-GCN training + explanation demo
│
├── src/
│   ├── build_kg.py          # CSV → NetworkX graph pipeline
│   ├── features.py          # node feature generator using sentence transformers
│   ├── pyg_loader.py        # load CSV → PyG tensors
│   ├── models.py            # R-GCN model class + scoring head
│   ├── rules.py             # fuzzy logic rule loss functions
│   └── explain.py           # simple explanation extraction utilities
│
├── rules/
│   └── rules.json           # 5–10 clinical rules for the neuro-symbolic layer
│
├── docs/
│   ├── architecture.md      # system diagram + rationale
│   ├── license_notes.md     # SNOMED compliance notes
│   └── meeting_script.md    # short pitch + demo walkthrough
│
├── requirements.txt
└── README.md
```

**Quick Start:**

1️⃣ To install dependencies

```bash
pip install -r requirements.txt
```

2️⃣ To build the Knowledge Graph using NetworkX

```bash
python src/build_kg.py \
    --entities data/entities.csv \
    --triples data/triples.csv \
    --out fig/kg_static.png
```

3️⃣ To generate Node Features using Bio/Clinical BERT

```bash
python src/features.py \
    --entities data/entities.csv \
    --out data/node_feats.npy
```

4️⃣ To train a minimal R-GCN demo:

```bash
python src/train_demo.py \
    --entities data/entities.csv \
    --triples data/triples.csv \
    --feats data/node_feats.npy
```

Outputs typically include:

● Node embeddings

● Link prediction scores

● Rule-loss values

● A minimal explanation trace (rule activation + path)

**Author:**

Aman Moon

MTech,

Indian Institute of Technology Kharagpur
