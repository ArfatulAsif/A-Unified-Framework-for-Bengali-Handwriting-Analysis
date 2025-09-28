#!/usr/bin/env python3
# /clustering/evaluate_clustering.py
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from page.page_embedding import load_patch_encoder, embed_page
from clustering.cluster import cluster_pages_dbscan

def set_seeds(seed: int):
    import random as pyrand
    np.random.seed(seed); pyrand.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def index_writer_pages(root: Path) -> Dict[str, List[Path]]:
    """Expect: root/<writer_id>/*.(png|jpg|jpeg|tif|bmp)"""
    writers: Dict[str, List[Path]] = {}
    for wdir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            imgs.extend(sorted(wdir.glob(ext)))
        if imgs:
            writers[wdir.name] = imgs
    return writers

def distance(e1: np.ndarray, e2: np.ndarray, metric: str) -> float:
    return 1.0 - float(np.dot(e1, e2)) if metric == "cosine" else float(np.linalg.norm(e1 - e2))

def score_from_distance(dist: float, metric: str) -> float:
    """Higher score = more likely same-writer (needed for AUC)."""
    return (1.0 - dist) if metric == "cosine" else (-dist)

def pairs_from_clusters(labels: np.ndarray) -> np.ndarray:
    """
    Build a binary matrix P where P[i,j]=1 iff labels[i]==labels[j] and label!=-1 (same cluster, not noise).
    """
    same = (labels[:, None] == labels[None, :])
    not_noise = (labels[:, None] != -1) & (labels[None, :] != -1)
    return (same & not_noise).astype(int)

def main():
    cfg = yaml.safe_load(open("config.yml", "r"))
    seed = cfg.get("runtime", {}).get("seed", 123)
    set_seeds(seed)

    metric = cfg["evaluation_page"]["metric"]
    # DBSCAN params (add to your config if you want; here we give sensible defaults)
    eps = float(cfg.get("clustering", {}).get("eps", 0.35))
    min_samples = int(cfg.get("clustering", {}).get("min_samples", 3))

    
    
    page_root = Path(cfg["paths"]["Test_page_data"])


    wmap = index_writer_pages(page_root)
    if not wmap:
        raise SystemExit(f"No page data found under {page_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_encoder = load_patch_encoder(cfg, device)

    # ---- Collect all pages and embed (with progress) ----
    all_items: List[Tuple[str, Path]] = []
    for wid, files in wmap.items():
        for p in files:
            all_items.append((wid, p))
    print(f"Found {len(all_items)} pages across {len(wmap)} writers.")

    embs: List[np.ndarray] = []
    y_writers: List[str] = []
    for wid, p in tqdm(all_items, desc="Embedding pages", unit="page"):
        e = embed_page(p, cfg, patch_encoder, device)
        if e is None:
            continue
        embs.append(e)
        y_writers.append(wid)

    if len(embs) < 2:
        raise SystemExit("Not enough embeddings for clustering.")

    E = np.vstack(embs)     # (N,D)
    N = E.shape[0]
    print(f"Embeddings ready: {N}")

    # ---- Cluster with DBSCAN ----
    labels, clusters, D = cluster_pages_dbscan(E, metric=metric, eps=eps, min_samples=min_samples)
    n_clusters = len(clusters)
    n_noise = int(np.sum(labels == -1))
    print(f"DBSCAN → clusters: {n_clusters}, noise pages: {n_noise}")

    # ---- Evaluate cluster quality as pair classification ----
    # Ground-truth pair labels: same writer?
    y_true_pairs: List[int] = []
    # Predictions from clustering: same cluster?
    y_pred_pairs: List[int] = []
    # Continuous score from distance (for AUC)
    y_score_pairs: List[float] = []

    # Precompute matrix for faster loop
    pred_same_mat = pairs_from_clusters(labels)  # (N,N) binary
    # distance matrix consistent with metric (use the one from clustering)
    # Convert to score (higher = more similar)
    if metric == "cosine":
        y_score_mat = 1.0 - D
    else:
        y_score_mat = -D

    # Iterate i<j pairs
    for i in range(N):
        for j in range(i + 1, N):
            same_writer = 1 if y_writers[i] == y_writers[j] else 0
            same_cluster = int(pred_same_mat[i, j])
            score_ij = float(y_score_mat[i, j])

            y_true_pairs.append(same_writer)
            y_pred_pairs.append(same_cluster)
            y_score_pairs.append(score_ij)

    y_true = np.asarray(y_true_pairs, dtype=int)
    y_pred = np.asarray(y_pred_pairs, dtype=int)
    y_score = np.asarray(y_score_pairs, dtype=float)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    print("\n--- Clustering Evaluation (pairwise) ---")
    print(f"Metric: {metric} | eps: {eps:.4f} | min_samples: {min_samples}")
    print(f"Clusters: {n_clusters} | Noise: {n_noise} | Pages: {N}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"FPR: {fpr:.4f} | FNR: {fnr:.4f}")

if __name__ == "__main__":
    main()