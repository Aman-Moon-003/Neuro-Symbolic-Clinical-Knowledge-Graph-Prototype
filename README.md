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

**Architecture Diagram:**


<img width="1003" height="421" alt="image" src="https://github.com/user-attachments/assets/5c861e26-8122-4e20-8913-85bb7f36817b" />



**Explanation Module Diagram:**


<img width="1060" height="547" alt="EXPLANATION_MODULE_ARCHITECTURE" src="https://github.com/user-attachments/assets/fa6f1112-72cf-413c-9ade-fbf1d181b289" />


**Repository Structure:**

```bash
├── data/
│   ├── out
│   │    └──pneumonia_1hop
│   │           ├── legend.png
│   │           ├── kg.gml
│   │           └── kg.png
│   ├── pyg_dataset
│   │        ├── summary.txt
│   │        ├── explanations.json
│   │        ├── test_triples.pt
│   │        ├── node_feats.npy
│   │        ├── train_triples.pt
│   │        └── valid_triples.pt
│   └── rules
│        └── rule_config.json
│
├── src/
│   ├── convert_nodes_edges.py
│   ├── validate_kg.py
│   ├── Filter_Relations.py
│   ├── visualize_kg.py
│   ├── safe_split.py
│   ├── create_node_features.py
│   ├── dataset.py
│   ├── update_dataset_features.py
│   ├── rule_engine.py
│   ├── train_rgcn.py
│   ├── explain.py
│   └── interactive_explain.py
│
├── docs/
│   └── ARCHITECTURE_DIAGRAM.png
│
├── EXPLANATION_MODULE_ARCHITECTURE.png
├── LICENSE.md
├── requirements.txt
└── README.md
```


**Outputs typically include:**

● Node embeddings

● Link prediction scores

● Rule-loss values

● A minimal explanation trace (rule activation + path)


