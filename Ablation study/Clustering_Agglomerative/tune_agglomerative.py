#!/usr/bin/env python3
# /Clustering_Agglomerative/tune_agglomerative.py




# python -m Clustering_Agglomerative.tune_agglomerative

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.cluster import AgglomerativeClustering

from page.page_embedding import load_patch_encoder, embed_page
from clustering.cluster import _pairwise_distance_matrix


def set_seeds(seed: int):
    import random as pyrand
    np.random.seed(seed); pyrand.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def index_writer_pages(root: Path) -> Dict[str, List[Path]]:
    writers: Dict[str, List[Path]] = {}
    for wdir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            imgs.extend(sorted(wdir.glob(ext)))
        if imgs:
            writers[wdir.name] = imgs
    return writers

def pairs_from_clusters(labels: np.ndarray) -> np.ndarray:
    same = (labels[:, None] == labels[None, :])
    # Agglomerative doesn't have noise (-1), so we just check for same cluster
    return same.astype(int)

def build_pairwise_ground_truth(writer_ids: List[str]) -> np.ndarray:
    w = np.array(writer_ids)
    return (w[:, None] == w[None, :]).astype(int)

def evaluate_pairwise(y_true_mat: np.ndarray, y_pred_mat: np.ndarray, score_mat: np.ndarray):
    n = y_true_mat.shape[0]
    y_true, y_pred, y_score = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            y_true.append(int(y_true_mat[i, j]))
            y_pred.append(int(y_pred_mat[i, j]))
            y_score.append(float(score_mat[i, j]))
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    y_score = np.asarray(y_score, float)

    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    return {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1,
        "fpr": fpr, "fnr": fnr, "auc": auc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def main():
    cfg = yaml.safe_load(open("config.yml", "r"))
    seed = cfg.get("runtime", {}).get("seed", 123)
    set_seeds(seed)

    metric = cfg["evaluation_page"]["metric"]
    default_thresh = float(cfg["evaluation_page"]["default_threshold"])

    tune_cfg = cfg.get("clustering_tuning", {}) or {}
    thresh_start = float(tune_cfg.get("eps_start", max(0.0, default_thresh - 0.10)))
    thresh_stop  = float(tune_cfg.get("eps_stop",  default_thresh + 0.11)) 
    thresh_step  = float(tune_cfg.get("eps_step",  0.01))
    
    # We tune on F1 by default to balance Precision and Recall
    objective = str(tune_cfg.get("objective", "f1")).lower()

    thresh_values = np.arange(thresh_start, thresh_stop, thresh_step)
    if thresh_values.size == 0:
        raise SystemExit("Empty threshold sweep range.")

    page_root = Path(cfg["paths"].get("page_test_dir") or cfg["paths"].get("Test_page_data"))
    wmap = index_writer_pages(page_root)
    if not wmap:
        raise SystemExit(f"No page data under: {page_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_encoder = load_patch_encoder(cfg, device)

    items: List[Tuple[str, Path]] = []
    for wid, files in wmap.items():
        for p in files:
            items.append((wid, p))
    print(f"Found {len(items)} pages across {len(wmap)} writers.")

    E_list: List[np.ndarray] = []
    writer_ids: List[str] = []
    for wid, p in tqdm(items, desc="Embedding pages", unit="page"):
        emb = embed_page(p, cfg, patch_encoder, device)
        if emb is None: continue
        E_list.append(emb)
        writer_ids.append(wid)

    E = np.vstack(E_list)
    N = E.shape[0]

    D = _pairwise_distance_matrix(E, metric=metric)
    score_mat = (1.0 - D) if metric == "cosine" else (-D)
    y_true_mat = build_pairwise_ground_truth(writer_ids)

    results = []
    print(f"Sweeping distance_threshold in [{thresh_start:.3f}, {thresh_stop:.3f}) step {thresh_step:.3f}")
    
    for thresh in thresh_values:
        agg = AgglomerativeClustering(n_clusters=None, distance_threshold=float(thresh), metric="precomputed", linkage="average")
        labels = agg.fit_predict(D)

        n_clusters = len(set(labels))
        
        ari = adjusted_rand_score(writer_ids, labels)
        nmi = normalized_mutual_info_score(writer_ids, labels)

        y_pred_mat = pairs_from_clusters(labels)
        metrics = evaluate_pairwise(y_true_mat, y_pred_mat, score_mat)

        results.append({
            "threshold": float(thresh),
            "clusters": int(n_clusters),
            "ari": float(ari),
            "nmi": float(nmi),
            **metrics,
        })

    if objective not in {"acc", "f1", "ari", "nmi"}:
        objective = "f1"
    best = max(results, key=lambda r: r[objective])

    print("\n=== Tuning Summary (top 10 by {}) ===".format(objective.upper()))
    top = sorted(results, key=lambda r: r[objective], reverse=True)[:10]
    for r in top:
        print(
            f"threshold={r['threshold']:.3f} | clusters={r['clusters']} | "
            f"ARI={r['ari']:.4f} NMI={r['nmi']:.4f} | "
            f"ACC={r['acc']:.4f} F1={r['f1']:.4f} PRE={r['precision']:.4f} REC={r['recall']:.4f}"
        )

    print("\n=== Best Setting (by {}) ===".format(best and objective.upper()))
    print(f"distance_threshold={best['threshold']:.6f} | clusters={best['clusters']}\n"
          f"ARI={best['ari']:.4f} | NMI={best['nmi']:.4f}\n"
          f"ACC={best['acc']:.4f} | PRE={best['precision']:.4f} | REC={best['recall']:.4f} | F1={best['f1']:.4f}")

if __name__ == "__main__":
    main()