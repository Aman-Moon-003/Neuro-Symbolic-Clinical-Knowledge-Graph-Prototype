#!/usr/bin/env python3
"""
Train a Relational Graph Convolutional Network (R-GCN) for link prediction
on your SNOMED-based clinical knowledge graph.

Usage:
    python src/train_rgcn.py \
        --dataset_dir data/pyg_dataset \
        --save_dir checkpoints/rgcn \
        --epochs 50 \
        --hidden_dim 128 \
        --lr 1e-3 \
        --neg_ratio 10

"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
import torch.nn.functional as F
from rule_engine import RuleEngine
from typing import Tuple


# -----------------------------------------------------------
#  R-GCN Model
# -----------------------------------------------------------
class RGCNLinkPredictor(nn.Module):
    def __init__(self, num_nodes, num_relations, in_dim, hidden_dim, num_layers=2):
        super().__init__()

        self.num_relations = num_relations
        self.emb = nn.Embedding(num_nodes, in_dim)

        self.layers = nn.ModuleList()
        self.layers.append(RGCNConv(in_dim, hidden_dim, num_relations))
        for _ in range(num_layers - 1):
            self.layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        # Score function: DistMult
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)

    def encode(self, data: Data):
        x = self.emb.weight
        edge_index = data.edge_index
        edge_type = data.edge_type

        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
            x = F.relu(x)

        return x

    def score(self, head, rel, tail):
        """
        DistMult scoring: <h * r, t>
        """
        h = head
        r = self.rel_emb(rel)
        t = tail
        return torch.sum(h * r * t, dim=-1)

    def forward(self, data: Data):
        return self.encode(data)


# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------
def load_triples(path: str):
    t = torch.load(path)
    return t[next(iter(t))]  # returns (h,r,t)


def negative_sampling(h, r, t, num_nodes, neg_ratio=10, device="cpu"):
    """
    Replace tails with random nodes.
    """
    n = h.size(0)
    neg_t = torch.randint(0, num_nodes, (n * neg_ratio,), device=device)
    neg_h = h.repeat_interleave(neg_ratio)
    neg_r = r.repeat_interleave(neg_ratio)
    return neg_h, neg_r, neg_t


def compute_metrics(model, data, triples, device="cpu"):
    """
    Compute MRR and Hits@K (1,3,10) for link prediction.
    """
    model.eval()
    h_idx, r_idx, t_idx = triples
    h_idx, r_idx, t_idx = h_idx.to(device), r_idx.to(device), t_idx.to(device)

    with torch.no_grad():
        x = model.encode(data)

    num_nodes = x.size(0)

    ranks = []
    for i in range(len(h_idx)):
        h = x[h_idx[i]]
        r = model.rel_emb(r_idx[i])

        # scores for all possible tails
        scores = torch.matmul(h * r, x.t())  # shape: (num_nodes,)
        target_score = scores[t_idx[i]]

        rank = (scores >= target_score).sum().item()
        ranks.append(rank)

    ranks = torch.tensor(ranks)
    mrr = (1.0 / ranks.float()).mean().item()
    hits1 = (ranks <= 1).float().mean().item()
    hits3 = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()

    return mrr, hits1, hits3, hits10


# -----------------------------------------------------------
# Training Loop
# -----------------------------------------------------------
def train_epoch(model, data, train_triples, optimizer, neg_ratio, rule_engine, lambda_rule=0.5, device="cpu"):

    model.train()

    h, r, t = train_triples
    h, r, t = h.to(device), r.to(device), t.to(device)

    x = model.encode(data)   # node embeddings [N, hidden_dim]

    pos_head, pos_rel, pos_tail = x[h], r, x[t]
    pos_scores = model.score(pos_head, pos_rel, pos_tail)
    pos_labels = torch.ones_like(pos_scores)

    # Negative sampling (unchanged)
    neg_h, neg_r, neg_t = negative_sampling(h, r, t, x.size(0), neg_ratio, device)
    neg_head, neg_rel, neg_tail = x[neg_h], neg_r, x[neg_t]
    neg_scores = model.score(neg_head, neg_rel, neg_tail)
    neg_labels = torch.zeros_like(neg_scores)

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    bce_loss = F.binary_cross_entropy_with_logits(scores, labels)

    # ---------- RULE LOSS ----------
    # compute symbolic rule loss from node embeddings (scalar tensor)
    rule_loss = rule_engine.compute_rule_loss(x)  # x is [N, hidden_dim]
    # combine
    loss = bce_loss + lambda_rule * rule_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # return numeric scalars for logging
    return loss.item(), bce_loss.item(), rule_loss.item()



# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\pyg_dataset")
    ap.add_argument("--save_dir", default="checkpoints/rgcn")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--neg_ratio", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lambda_rule", type=float, default=0.5, help="weight for rule loss")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device
    print(f"[INFO] Using device: {device}")

    # Load Data
    print("[INFO] Loading dataset...")
    data_path = os.path.join(args.dataset_dir, "data.pth")
    data_obj = torch.load(data_path, map_location="cpu", weights_only=False)

    if data_obj["data_obj"] is not None:
        data = data_obj["data_obj"].to(device)
    else:
        from torch_geometric.data import Data
        data = Data(x=data_obj['x'].to(device), edge_index=data_obj['edge_index'].to(device), edge_type=data_obj['edge_type'].to(device))

    x = data_obj["x"].to(device)
    data.x = x  # override features

    train_triples = load_triples(os.path.join(args.dataset_dir, "train_triples.pt"))
    valid_triples = load_triples(os.path.join(args.dataset_dir, "valid_triples.pt"))
    test_triples  = load_triples(os.path.join(args.dataset_dir, "test_triples.pt"))

    # Load mappings needed for model and rule engine
    with open(os.path.join(args.dataset_dir, "rel2id.json")) as f:
        rel2id = json.load(f)

    # Load id2idx mapping (needed for rule engine)
    with open(os.path.join(args.dataset_dir, "id2idx.json")) as f:
        id2idx = json.load(f)

    # Initialize RuleEngine (learnable). emb_dim should equal the model output embedding dim.
    # model.encode(...) outputs vectors of size hidden_dim, so set emb_dim = args.hidden_dim
    rule_engine = RuleEngine(
        rule_config_path=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\rules\rule_config.json",
        id2idx=id2idx,
        rel2id=rel2id,
        emb_dim=args.hidden_dim,
        device=device
    )

    num_nodes = data.x.size(0)
    num_relations = len(rel2id)
    in_dim = data.x.size(1)

    # Build model
    model = RGCNLinkPredictor(
        num_nodes=num_nodes,
        num_relations=num_relations,
        in_dim=in_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    # include rule_engine parameters for joint training
    optimizer = optim.Adam(
        list(model.parameters()) + list(rule_engine.parameters()),
        lr=args.lr
    )

    # Training Loop
    print("[INFO] Starting training...")

    for epoch in range(1, args.epochs + 1):
        total_loss, bce_l, rule_l = train_epoch(model, data, train_triples, optimizer, args.neg_ratio, rule_engine,
                                                lambda_rule=0.5, device=device)

        if epoch % 5 == 0:
            mrr, h1, h3, h10 = compute_metrics(model, data, valid_triples, device)
            print(
                f"[EPOCH {epoch}] TotalLoss={total_loss:.4f}  BCE={bce_l:.4f}  RuleLoss={rule_l:.4f}  Val MRR={mrr:.4f}  Hits@10={h10:.4f}")

            # Save combined checkpoint (save both model and rule engine weights)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "rule_state_dict": rule_engine.state_dict(),
                "args": vars(args)
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"rgcn_epoch{epoch}.pt"))

    print("[INFO] Training complete.")

    # Final Test Evaluation
    print("[INFO] Evaluating on TEST set...")
    mrr, h1, h3, h10 = compute_metrics(model, data, test_triples, device)
    print(f"[TEST] MRR={mrr:.4f} Hits@1={h1:.4f} Hits@3={h3:.4f} Hits@10={h10:.4f}")


if __name__ == "__main__":
    main()
