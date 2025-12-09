import torch
import numpy as np
import os
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\data\pyg_dataset", help="Folder with data.pth")
    ap.add_argument("--features", default=r"C:\Users\AMAN\Documents\MTech Thesis Projects\Neuro-Symbolic Clinical Knowledge Graph Prototype\src\data\converted\node_feats.npy", help="node_feats.npy path")
    ap.add_argument("--save_as", default="data_with_features.pth")
    args = ap.parse_args()

    data_pth = os.path.join(args.dataset_dir, "data.pth")
    print("[INFO] Loading:", data_pth)
    data = torch.load(data_pth, weights_only=False)

    feats = np.load(args.features)
    feats = torch.tensor(feats, dtype=torch.float)

    if data['x'].shape[0] != feats.shape[0]:
        raise ValueError(f"Node count mismatch: data.x={data['x'].shape}, feats={feats.shape}")

    print(f"[INFO] Replacing data.x with new features {feats.shape}")
    data['x'] = feats

    save_path = os.path.join(args.dataset_dir, args.save_as)
    torch.save(data, save_path)
    print("[OK] Saved updated dataset to:", save_path)

if __name__ == "__main__":
    main()
