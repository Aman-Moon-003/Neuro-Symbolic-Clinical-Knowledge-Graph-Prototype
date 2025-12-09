# src/rule_engine.py
import json
import torch
import torch.nn as nn

# fuzzy ops (Łukasiewicz)
def l_and(x, y):
    return torch.clamp(x + y - 1.0, min=0.0)

def l_not(x):
    return 1.0 - x

def l_implies(x, y):
    return torch.clamp(1.0 - x + y, max=1.0)


class RuleEngine(nn.Module):
    """
    Learnable rule engine: projects node embeddings -> scalar truth-value via a small linear readout.
    Expects node_emb: tensor [N, emb_dim]
    rule_config_path: path to JSON with rules (see format in earlier message).
    id2idx / rel2id: dict mappings (string keys).
    """
    def __init__(self, rule_config_path, id2idx, rel2id, emb_dim, device="cpu"):
        super().__init__()
        self.id2idx = {str(k): int(v) for k, v in id2idx.items()}
        self.rel2id = rel2id
        self.device = device

        with open(rule_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.rules = cfg.get("rules", [])

        # readout: embedding_dim -> scalar logit
        self.readout = nn.Linear(emb_dim, 1)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.1)
        nn.init.constant_(self.readout.bias, 0.0)

        # move to device when created/loaded in train script
        self.to(device)

    def get_concept_score(self, node_emb, concept_id):
        idx = self.id2idx[str(concept_id)]
        emb = node_emb[idx].to(self.device)      # [D]
        logit = self.readout(emb)               # [1]
        prob = torch.sigmoid(logit).squeeze()   # scalar
        return prob

    def get_relation_score(self, node_emb, head_id, rel_name=None):
        # For simplicity use same readout on head embedding
        idx = self.id2idx[str(head_id)]
        emb = node_emb[idx].to(self.device)
        logit = self.readout(emb)
        return torch.sigmoid(logit).squeeze()

    def compute_rule_loss(self, node_emb):
        """
        node_emb: [N, D] tensor
        returns: scalar tensor (loss)
        """
        total = torch.tensor(0.0, device=self.device)
        for rule in self.rules:
            wt = float(rule.get("weight", 1.0))
            rtype = rule.get("type", "implies")

            if rtype == "implies":
                prem = self.get_relation_score(node_emb, rule["premise"]["concept"], rule["premise"].get("relation", None))
                conc = self.get_concept_score(node_emb, rule["conclusion"]["concept"])
                loss = 1.0 - l_implies(prem, conc)
                total = total + wt * loss

            elif rtype == "and_implies":
                pvals = []
                for p in rule["premise"]:
                    pvals.append(self.get_concept_score(node_emb, p["concept"]))
                prem = pvals[0]
                for v in pvals[1:]:
                    prem = l_and(prem, v)
                conc = self.get_concept_score(node_emb, rule["conclusion"]["concept"])
                loss = 1.0 - l_implies(prem, conc)
                total = total + wt * loss

            elif rtype == "implies_not":
                prem = self.get_concept_score(node_emb, rule["premise"]["concept"])
                conc = l_not(self.get_concept_score(node_emb, rule["conclusion"]["concept"]))
                loss = 1.0 - l_implies(prem, conc)
                total = total + wt * loss

            else:
                raise ValueError(f"Unknown rule type: {rtype}")

        return total
