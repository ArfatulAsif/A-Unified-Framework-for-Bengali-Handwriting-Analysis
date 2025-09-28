#!/usr/bin/env python3
# /clustering/cluster.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

def _pairwise_distance_matrix(embs: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    embs: (N, D) L2-normalized recommended
    metric: "cosine" or "l2"
    returns D: (N, N) distances (smaller = more similar)
    """
    if metric == "cosine":
        # cosine distance = 1 - cosine similarity
        S = embs @ embs.T
        S = np.clip(S, -1.0, 1.0)
        return 1.0 - S
    elif metric.lower() in ("l2", "euclidean"):
        # ||a-b||_2 using (a-b)^2 = a^2 + b^2 - 2ab
        sq = np.sum(embs**2, axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * (embs @ embs.T)
        D2 = np.maximum(D2, 0.0)
        return np.sqrt(D2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def cluster_pages_dbscan(
    embs: np.ndarray,
    metric: str = "cosine",
    eps: float = 0.35,
    min_samples: int = 3,
) -> Tuple[np.ndarray, Dict[int, List[int]], np.ndarray]:
    """
    Cluster page embeddings with DBSCAN on a precomputed distance matrix.

    Returns:
      labels: (N,) cluster labels; -1 indicates noise
      clusters: {cluster_id: [indices...]}, cluster_id excludes -1
      dist_matrix: (N,N) distances used for clustering
    """
    D = _pairwise_distance_matrix(embs, metric=metric)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(D)

    clusters: Dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        if c == -1:
            continue
        clusters.setdefault(c, []).append(i)

    return labels, clusters, D