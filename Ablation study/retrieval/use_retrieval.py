
# /retrieval/use_retrieval.py


"""
# Use retrieval: copy same-writer pages to out folder
python -m retrieval.use_retrieval /path/to/pool /path/to/reference_page.jpg /path/to/save_dir


# (optional) include the reference page in results
python -m retrieval.use_retrieval /path/to/pool /path/to/reference_page.jpg /path/to/save_dir --include-ref






python -m retrieval.use_retrieval ./data/Different_data_set_for_retrieval_and_clustering_testing ./data/Different_data_set_for_retrieval_and_clustering_testing/0002_02.tif ./retrieval/output_of_use_retrieval


python -m retrieval.use_retrieval ./data/Different_data_set_for_retrieval_and_clustering_testing ./data/Different_data_set_for_retrieval_and_clustering_testing/0001_01.tif ./retrieval/output_of_use_retrieval


"""



from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from tqdm import tqdm

from page.page_embedding import load_patch_encoder, embed_page


def distance(e1: np.ndarray, e2: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return 1.0 - float(np.dot(e1, e2))
    else:
        return float(np.linalg.norm(e1 - e2))


def collect_images(root: Path) -> List[Path]:
    """
    Recursively collect images under root. Works whether or not the pool is organized by writer.
    """
    paths: List[Path] = []
    for ext in ("**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.bmp", "**/*.tif", "**/*.tiff"):
        paths.extend(root.glob(ext))
    # Remove duplicates & ensure deterministic order
    paths = sorted(set(p.resolve() for p in paths))
    return paths


def main():
    ap = argparse.ArgumentParser(description="Retrieve all pages by the same writer as the reference page.")
    ap.add_argument("pool_root", help="Path to a folder containing page images (recursively scanned).")
    ap.add_argument("reference_page", help="Path to a single reference page image.")
    ap.add_argument("save_dir", help="Destination folder to copy retrieved pages into.")
    ap.add_argument("--config", default="config.yml", help="Path to config.yml (for metric & threshold).")
    ap.add_argument("--include-ref", action="store_true", help="Include the reference page in the copied results.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    metric = cfg["evaluation_page"]["metric"]
    threshold = float(cfg["evaluation_page"]["default_threshold"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_encoder = load_patch_encoder(cfg, device)

    pool_root = Path(args.pool_root).resolve()
    ref_page = Path(args.reference_page).resolve()
    out_dir = Path(args.save_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Embed reference page
    ref_emb = embed_page(ref_page, cfg, patch_encoder, device)
    if ref_emb is None:
        raise SystemExit(f"Failed to embed reference page (no lines detected?): {ref_page}")

    # Collect pool images
    pool_imgs = collect_images(pool_root)
    if not pool_imgs:
        raise SystemExit(f"No images found under: {pool_root}")

    print(f"Pool images found: {len(pool_imgs)}")
    kept = 0

    for img_path in tqdm(pool_imgs, desc="Scanning pool", unit="img"):
        if (not args.include_ref) and (img_path == ref_page):
            continue

        emb = embed_page(img_path, cfg, patch_encoder, device)
        if emb is None:
            continue
        dist = distance(ref_emb, emb, metric)
        if dist < threshold:
            # Copy into save_dir, keep filename (avoid collisions by prefixing with parent folder)
            dst_name = f"{img_path.parent.name}__{img_path.name}"
            shutil.copy2(str(img_path), str(out_dir / dst_name))
            kept += 1

    print(f"Copied {kept} pages to: {out_dir}")


if __name__ == "__main__":
    main()