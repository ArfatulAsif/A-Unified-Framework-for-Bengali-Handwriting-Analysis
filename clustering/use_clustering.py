
# /clustering/use_clustering.py


"""
python -m clustering.use_clustering /path/to/pool_images /path/to/save_output




python -m clustering.use_clustering ./data/Different_data_set_for_retrieval_and_clustering_testing ./clustering/output_of_use_clustering



"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import yaml
from tqdm import tqdm

from page.page_embedding import load_patch_encoder, embed_page
from clustering.cluster import cluster_pages_dbscan

def collect_images(root: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in ("**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.bmp", "**/*.tif", "**/*.tiff"):
        paths.extend(root.glob(ext))
    return sorted(set(p.resolve() for p in paths))

def main():
    ap = argparse.ArgumentParser(description="Cluster page images by handwriting and copy into per-cluster folders.")
    ap.add_argument("pool_root", help="Folder containing page images (recursively scanned).")
    ap.add_argument("save_root", help="Destination folder; clusters are created as subfolders.")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--eps", type=float, default=None, help="DBSCAN eps; overrides config if given.")
    ap.add_argument("--min-samples", type=int, default=None, help="DBSCAN min_samples; overrides config if given.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    metric = cfg["evaluation_page"]["metric"]
    eps = float(args.eps if args.eps is not None else cfg.get("clustering", {}).get("eps", 0.35))
    min_samples = int(args.min_samples if args.min_samples is not None else cfg.get("clustering", {}).get("min_samples", 3))

    pool_root = Path(args.pool_root).resolve()
    out_root = Path(args.save_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    imgs = collect_images(pool_root)
    if not imgs:
        raise SystemExit(f"No images found under: {pool_root}")
    print(f"Images found: {len(imgs)}")

    # Embed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_encoder = load_patch_encoder(cfg, device)

    embs: List[np.ndarray] = []
    kept_paths: List[Path] = []
    for p in tqdm(imgs, desc="Embedding pages", unit="img"):
        e = embed_page(p, cfg, patch_encoder, device)
        if e is None:
            continue
        embs.append(e); kept_paths.append(p)

    if len(embs) < 2:
        raise SystemExit("Not enough embeddings for clustering.")

    E = np.vstack(embs)
    labels, clusters, _ = cluster_pages_dbscan(E, metric=metric, eps=eps, min_samples=min_samples)
    n_clusters = len(clusters); n_noise = int(np.sum(labels == -1))
    print(f"DBSCAN → clusters: {n_clusters}, noise: {n_noise}")

    # Copy files by cluster
    # Map each label to folder name; -1 -> noise
    label_to_dir: Dict[int, Path] = {}
    for i, (p, c) in enumerate(zip(kept_paths, labels)):
        if c == -1:
            d = label_to_dir.setdefault(-1, out_root / "noise")
        else:
            d = label_to_dir.setdefault(c, out_root / f"cluster_{c:03d}")
        d.mkdir(parents=True, exist_ok=True)
        # avoid filename collisions by prefixing parent folder name
        dst = d / f"{p.parent.name}__{p.name}"
        shutil.copy2(str(p), str(dst))

    print(f"Clusters saved under: {out_root}")

if __name__ == "__main__":
    main()