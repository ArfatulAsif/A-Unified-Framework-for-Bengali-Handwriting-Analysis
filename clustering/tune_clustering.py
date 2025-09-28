
# /clustering/tune_clustering.py



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
)
from sklearn.cluster import DBSCAN

from page.page_embedding import load_patch_encoder, embed_page
from clustering.cluster import _pairwise_distance_matrix  # reuse same distance impl


def set_seeds(seed: int):
    import random as pyrand
    np.random.seed(seed); pyrand.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def index_writer_pages(root: Path) -> Dict[str, List[Path]]:
    """Expect: root/<writer_id>/*.(png|jpg|jpeg|bmp|tif|tiff)"""
    writers: Dict[str, List[Path]] = {}
    for wdir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            imgs.extend(sorted(wdir.glob(ext)))
        if imgs:
            writers[wdir.name] = imgs
    return writers


def pairs_from_clusters(labels: np.ndarray) -> np.ndarray:
    """Binary matrix P where P[i,j]=1 iff i & j are in the SAME (non-noise) cluster."""
    same = (labels[:, None] == labels[None, :])
    not_noise = (labels[:, None] != -1) & (labels[None, :] != -1)
    return (same & not_noise).astype(int)


def build_pairwise_ground_truth(writer_ids: List[str]) -> np.ndarray:
    w = np.array(writer_ids)
    return (w[:, None] == w[None, :]).astype(int)


def evaluate_pairwise(y_true_mat: np.ndarray, y_pred_mat: np.ndarray, score_mat: np.ndarray):
    """Turn matrices into pair lists and compute metrics."""
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
    # ---- Load config & defaults (no CLI args) ----
    cfg = yaml.safe_load(open("config.yml", "r"))
    seed = cfg.get("runtime", {}).get("seed", 123)
    set_seeds(seed)

    metric = cfg["evaluation_page"]["metric"]
    default_eps = float(cfg["evaluation_page"]["default_threshold"])

    # Optional config block to customize the sweep (no CLI needed)
    tune_cfg = cfg.get("clustering_tuning", {}) or {}
    eps_start = float(tune_cfg.get("eps_start", max(0.0, default_eps - 0.10)))
    eps_stop  = float(tune_cfg.get("eps_stop",  default_eps + 0.11))  # exclusive
    eps_step  = float(tune_cfg.get("eps_step",  0.01))
    min_samples_values = list(tune_cfg.get("min_samples", [2, 3]))
    objective = str(tune_cfg.get("objective", "acc")).lower()  # "acc" or "f1"

    eps_values = np.arange(eps_start, eps_stop, eps_step)
    if eps_values.size == 0:
        raise SystemExit("Empty eps sweep range from config.yml (clustering_tuning).")

    
    page_root = Path(cfg["paths"].get("page_test_dir") or cfg["paths"].get("Test_page_data"))


    if (page_root is None) or (not Path(page_root).exists()):
        raise SystemExit("Missing or invalid paths.page_test_dir (or paths.Test_page_data) in config.yml")

    wmap = index_writer_pages(page_root)
    if not wmap:
        raise SystemExit(f"No page data under: {page_root}")

    # ---- Embed once ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_encoder = load_patch_encoder(cfg, device)

    items: List[Tuple[str, Path]] = []
    for wid, files in wmap.items():
        for p in files:
            items.append((wid, p))
    print(f"Found {len(items)} pages across {len(wmap)} writers.")
    if len(items) < 2:
        raise SystemExit("Need at least 2 pages.")

    E_list: List[np.ndarray] = []
    writer_ids: List[str] = []
    for wid, p in tqdm(items, desc="Embedding pages", unit="page"):
        emb = embed_page(p, cfg, patch_encoder, device)
        if emb is None:
            continue
        E_list.append(emb)
        writer_ids.append(wid)

    if len(E_list) < 2:
        raise SystemExit("Not enough valid embeddings for tuning.")

    E = np.vstack(E_list)
    N = E.shape[0]
    print(f"Embeddings ready: {N}")

    # ---- Distance & score matrices (once) ----
    D = _pairwise_distance_matrix(E, metric=metric)
    score_mat = (1.0 - D) if metric == "cosine" else (-D)  # higher = more similar
    y_true_mat = build_pairwise_ground_truth(writer_ids)

    # ---- Sweep DBSCAN settings ----
    results = []
    print(f"Sweeping eps in [{eps_start:.3f}, {eps_stop:.3f}) step {eps_step:.3f}, min_samples in {min_samples_values}")
    for ms in min_samples_values:
        for eps in eps_values:
            db = DBSCAN(eps=float(eps), min_samples=int(ms), metric="precomputed")
            labels = db.fit_predict(D)

            n_clusters = len({c for c in labels if c != -1})
            n_noise = int(np.sum(labels == -1))

            y_pred_mat = pairs_from_clusters(labels)
            metrics = evaluate_pairwise(y_true_mat, y_pred_mat, score_mat)

            results.append({
                "eps": float(eps),
                "min_samples": int(ms),
                "clusters": int(n_clusters),
                "noise": int(n_noise),
                **metrics,
            })

    # ---- Choose best by objective ----
    if objective not in {"acc", "f1"}:
        objective = "acc"
    best = max(results, key=lambda r: r[objective])

    # ---- Print summary ----
    print("\n=== Tuning Summary (top 10 by {}) ===".format(objective.upper()))
    top = sorted(results, key=lambda r: r[objective], reverse=True)[:10]
    for r in top:
        print(
            f"eps={r['eps']:.3f} ms={r['min_samples']} | "
            f"clusters={r['clusters']} noise={r['noise']} | "
            f"ACC={r['acc']:.4f} F1={r['f1']:.4f} PRE={r['precision']:.4f} REC={r['recall']:.4f} "
            f"FPR={r['fpr']:.4f} FNR={r['fnr']:.4f} AUC={r['auc']:.4f}"
        )

    print("\n=== Best Setting (by {}) ===".format(best and objective.upper()))
    print(
        f"eps={best['eps']:.6f}, min_samples={best['min_samples']} | clusters={best['clusters']}, noise={best['noise']}\n"
        f"ACC={best['acc']:.4f} | PRE={best['precision']:.4f} | REC={best['recall']:.4f} | "
        f"F1={best['f1']:.4f} | FPR={best['fpr']:.4f} | FNR={best['fnr']:.4f} | AUC={best['auc']:.4f}"
    )

    print("\nSuggested config.yml update:")
    print(f"clustering:\n  eps: {best['eps']:.6f}\n  min_samples: {best['min_samples']}")


if __name__ == "__main__":
    main()