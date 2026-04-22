# /retrieval/evaluate_retrieval.py

# python -m retrieval.evaluate_retrieval



from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Use your page pipeline (this calls page.segment_lines under the hood)
from page.page_embedding import load_patch_encoder, embed_page


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


def calculate_average_precision(ranked_labels: List[int], total_relevant: int) -> float:
    """Calculates Average Precision (AP) for a single query."""
    if total_relevant == 0:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    for i, label in enumerate(ranked_labels):
        if label == 1:
            hits += 1
            sum_precisions += hits / (i + 1.0)
            
    return sum_precisions / total_relevant


def main():
    # -------- Load config & setup --------
    cfg = yaml.safe_load(open("config.yml", "r"))
    seed = cfg.get("runtime", {}).get("seed", 123)
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = cfg["evaluation_page"]["metric"]

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
            # Skip pages with no lines detected
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

    # -------- Retrieval evaluation (Leave-One-Out Ranking) --------
    print(f"\nRunning Full Database Retrieval (Queries: {len(pool)})...")
    
    top1_hits = 0
    top5_hits = 0
    aps = []

    # Use every single page as a search query exactly once
    for ref_idx in tqdm(range(len(pool)), desc="Ranking Queries", unit="query"):
        ref_writer, ref_path, ref_emb = pool[ref_idx]
        
        # Determine how many true matches exist in the database (excluding the query itself)
        total_relevant = sum(1 for w, p, _ in pool if w == ref_writer and p != ref_path)
        
        # If this writer only has 1 page in the whole dataset, we can't do retrieval for them
        if total_relevant == 0:
            continue 
            
        # Compare query against all OTHER pages in the database
        distances = []
        for tgt_idx, (tgt_writer, tgt_path, tgt_emb) in enumerate(pool):
            if ref_idx == tgt_idx:
                continue # Skip comparing the query to its exact self
                
            dist = distance(ref_emb, tgt_emb, metric)
            is_match = 1 if tgt_writer == ref_writer else 0
            distances.append((dist, is_match))
            
        # Sort database by distance (lowest distance first)
        distances.sort(key=lambda x: x[0])
        
        # Extract just the binary labels (1 for match, 0 for distractors) of the sorted list
        ranked_labels = [match for _, match in distances]
        
        # Top-1 Accuracy: Is the #1 closest result a true match?
        if ranked_labels[0] == 1:
            top1_hits += 1
            
        # Top-5 Accuracy: Is there at least one true match in the top 5 closest results?
        if sum(ranked_labels[:5]) > 0:
            top5_hits += 1
            
        # Calculate Average Precision for this specific query
        ap = calculate_average_precision(ranked_labels, total_relevant)
        aps.append(ap)

    # -------- Final Aggregated Metrics --------
    valid_queries = len(aps)
    top1_acc = top1_hits / valid_queries
    top5_acc = top5_hits / valid_queries
    mAP = np.mean(aps)

    print("\n" + "="*50)
    print(" RETRIEVAL EVALUATION (Ranking Based) ")
    print("="*50)
    print(f"Total Valid Queries : {valid_queries}")
    print(f"Distance Metric     : {metric}")
    print("-" * 50)
    print(f"Top-1 Accuracy      : {top1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy      : {top5_acc * 100:.2f}%")
    print(f"mAP                 : {mAP * 100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()