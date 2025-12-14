# Neuro-Symbolic-Clinical-Knowledge-Graph-Prototype
This repository is a prototype created as a component of a MTech thesis at IIT Kharagpur. It examines neuro-symbolic reasoning over clinical knowledge graphs with a focus on repeatability, interpretability, and logical consistency. It is designed for controlled experimentation rather than production deployment, the solution is purposefully simple and modular.


**Scope and Contribution:**

This prototype illustrates:

● Building a therapeutically matched knowledge graph with SNOMED-CT concepts

● Reasoning across multi-relational clinical graphs using Relational Graph Convolutional Networks (R-GCN)

● Using Łukasiewicz t-norms as soft logical constraints to integrate fuzzy clinical rules

● Creation of human-readable explanation trails that connect graph routes and rules to forecasts

The system is intended to investigate how neural reasoning in clinical knowledge graphs is influenced by symbolic limitations.


**Research Goals:**

This prototype supports the thesis:

**“Neuro-Symbolic AI for Clinical Knowledge Graph Reasoning.”**

The objectives are:

● Create a knowledge graph based on SNOMED-CT that has clinical significance.

● Use R-GCN or similar GNNs to learn relational representations.

● Use differentiable fuzzy rules to enforce clinical reasoning

● Boost interpretability and logical coherence


**SNOMED CT Licensing Notes:**

This repository does not include:

● SNOMED-CT RF2 files

● Official SNOMED-CT distributions

● Redistributable SNOMED content

All included data consists solely of:

● Hand-curated concept identifiers and labels

● Synthetic or manually constructed edges

● Research-only artifacts for prototyping

Upon approval of SNOMED-CT RF2 access, the current KG will be replaced with an RF2-derived knowledge graph.


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
│   ├── EXPLANATION_MODULE_ARCHITECTURE.png
│   └── ARCHITECTURE_DIAGRAM.png
│
├── LICENSE
├── requirements.txt
└── README.md
```


**How to run (minimal):**

1. Environment Setup:
   ```bash
   git clone <repository-url>
   cd Neuro-Symbolic-Clinical-Knowledge-Graph-Prototype
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   
2. Visualize the Knowledge Graph(KG):
   ```bash
   python src/validate_kg.py
   ```
   Checks structural consistency of nodes, relations, and edge types.
   
3. Prepare the dataset:
   ```bash
   python src/create_node_features.py
   python src/safe_split.py
   ```
   Generates node features and reproducible train/validation/test splits.
   
4. Train the R-GCN model:
   ```bash
   python src/train_rgcn.py
   ```
   Trains an R-GCN with integrated fuzzy rule loss.
   
5. Generate explanations:
    ```bash
    python src/explain.py
   ```
   Produces minimal explanation traces, including activated rules and supporting KG paths.
    

**Outputs:**

● Acquired knowledge about node embeddings

● Scores for link prediction

● Statistics on rule-violation (rule-loss)

● Explanation traces (graph pathways + rules)

Outputs are kept in data/pyg_dataset/


**Design Philosophy:**

● Research-first: Prioritize research over engineering complexity to ensure clarity and reproducibility.

● Interpretable: Clear guidelines and traces of explanation

● Modular: Parts can be added or changed on their own


Clinical deployment is not the intended use of this repository.
