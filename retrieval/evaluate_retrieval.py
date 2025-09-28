# /retrieval/evaluate_retrieval.py


from __future__ import annotations

import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm, trange
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Use your page pipeline (this calls page.segment_lines under the hood)
from page.page_embedding import load_patch_encoder, embed_page


# -------- Configurable loop count for retrieval evaluation --------
N_TRIALS = 20  # repeat retrieval with random references
# -----------------------------------------------------------------


def set_seeds(seed: int):
    import random as pyrand

    np.random.seed(seed)
    pyrand.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def index_writer_pages(root: Path) -> Dict[str, List[Path]]:
    """
    Expect: root/<writer_id>/*.(png|jpg|jpeg|tif|bmp)
    Returns: {writer_id: [page_paths...]}
    """
    writers: Dict[str, List[Path]] = {}
    for wdir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            imgs.extend(sorted(wdir.glob(ext)))
        if imgs:
            writers[wdir.name] = imgs
    return writers


def distance(e1: np.ndarray, e2: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return 1.0 - float(np.dot(e1, e2))
    else:
        return float(np.linalg.norm(e1 - e2))


def score_from_distance(dist: float, metric: str) -> float:
    """
    AUC needs a score where higher = more likely positive (same writer).
    For cosine distance: score = 1 - dist ∈ [-∞, 1], higher is more similar.
    For L2: use negative distance so higher is more similar.
    """
    return (1.0 - dist) if metric == "cosine" else (-dist)


def main():
    # -------- Load config & setup --------
    cfg = yaml.safe_load(open("config.yml", "r"))
    seed = cfg.get("runtime", {}).get("seed", 123)
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = cfg["evaluation_page"]["metric"]
    threshold = float(cfg["evaluation_page"]["default_threshold"])

    test_root = Path(cfg["paths"]["Test_page_data"])
    wmap = index_writer_pages(test_root)
    if not wmap:
        raise SystemExit(f"No page data found under: {test_root}")

    patch_encoder = load_patch_encoder(cfg, device)

    # -------- Embed all pages once (with progress) --------
    # Build: writer -> list[(path, embedding)]
    writer_embs: Dict[str, List[Tuple[Path, np.ndarray]]] = {}
    all_pages: List[Tuple[str, Path]] = []  # (writer_id, path)

    for wid, files in wmap.items():
        all_pages.extend((wid, p) for p in files)

    print(f"Found {len(all_pages)} total pages across {len(wmap)} writers.")
    print("Embedding pages (this uses page.segment_lines internally)...")

    cache: Dict[Path, np.ndarray] = {}

    def get_embed(p: Path) -> np.ndarray | None:
        if p in cache:
            return cache[p]
        emb = embed_page(p, cfg, patch_encoder, device)
        cache[p] = emb
        return emb

    for wid in wmap:
        writer_embs[wid] = []
    for wid, p in tqdm(all_pages, desc="Embedding pages", unit="page"):
        emb = get_embed(p)
        if emb is None:
            # Optional: skip pages with no lines detected
            continue
        writer_embs[wid].append((p, emb))

    # Flatten pool of (writer, path, emb) for retrieval
    pool: List[Tuple[str, Path, np.ndarray]] = []
    for wid, lst in writer_embs.items():
        for p, e in lst:
            pool.append((wid, p, e))

    if len(pool) < 2:
        raise SystemExit("Not enough embedded pages to evaluate retrieval.")

    print(f"Page pool size (embedded): {len(pool)}")

    # -------- Retrieval evaluation over multiple trials --------
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_score_all: List[float] = []

    rng = random.Random(seed)

    print(f"Running {N_TRIALS} retrieval trials...")
    for _ in trange(N_TRIALS, desc="Retrieval trials", unit="trial"):
        # Pick a random reference page
        ref_idx = rng.randrange(len(pool))
        ref_writer, ref_path, ref_emb = pool[ref_idx]

        # Compare with every other page in the pool (exclude the same file)
        for tgt_writer, tgt_path, tgt_emb in pool:
            if tgt_path == ref_path:
                continue

            dist = distance(ref_emb, tgt_emb, metric)
            score = score_from_distance(dist, metric)

            pred = 1 if dist < threshold else 0
            
            label = 1 if tgt_writer == ref_writer else 0

            y_true_all.append(label)
            y_pred_all.append(pred)
            y_score_all.append(score)

    y_true = np.asarray(y_true_all, dtype=int)
    y_pred = np.asarray(y_pred_all, dtype=int)
    y_score = np.asarray(y_score_all, dtype=float)

    # -------- Metrics --------
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    print("\n--- Retrieval Evaluation (page-level) ---")
    print(f"Trials: {N_TRIALS} | Pool size: {len(pool)} | Threshold: {threshold:.6f} | Metric: {metric}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"FPR: {fpr:.4f} | FNR: {fnr:.4f}")


if __name__ == "__main__":
    main()