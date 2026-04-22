# /page/predict.py



"""
python -m page.predict ./data/Test_For_Pages/231/231_1.jpg ./data/Test_For_Pages/232/232_1.jpg



python -m page.predict ./data/Test_For_Pages/231/231_1.jpg ./data/Test_For_Pages/231/231_2.jpg


python -m page.predict ./data/Test_For_Pages/232/232_1.jpg ./data/Test_For_Pages/232/232_2.jpg


python -m page.predict ./data/Different_data_set_for_retrieval_and_clustering_testing/0001_01.tif ./data/Different_data_set_for_retrieval_and_clustering_testing/0000_02.tif



python -m page.predict ./data/Different_data_set_for_retrieval_and_clustering_testing/0001_01.tif ./data/Different_data_set_for_retrieval_and_clustering_testing/0001_03.tif


"""


from __future__ import annotations
import argparse
import yaml
import numpy as np
import torch
from page.page_embedding import load_patch_encoder, embed_page

def distance(e1: np.ndarray, e2: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return 1.0 - float(np.dot(e1, e2))
    else:
        return float(np.linalg.norm(e1 - e2))

def main():
    ap = argparse.ArgumentParser(description="Predict same/different writer for two pages.")
    ap.add_argument("page1")
    ap.add_argument("page2")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decision threshold; defaults to config evaluation_page.default_threshold")
    ap.add_argument("--config", default="config.yml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    metric = cfg["evaluation_page"]["metric"]
    th = args.threshold if args.threshold is not None else float(cfg["evaluation_page"]["default_threshold"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    patch_encoder = load_patch_encoder(cfg, device)

    e1 = embed_page(args.page1, cfg, patch_encoder, device)
    e2 = embed_page(args.page2, cfg, patch_encoder, device)

    if e1 is None or e2 is None:
        print("Failed to extract page embeddings (no lines found).")
        raise SystemExit(2)

    dist = distance(e1, e2, metric)
    label = "same writer" if dist < th else "different writer"

    print(f"distance={dist:.6f} | threshold={th:.6f} | decision={label}")

if __name__ == "__main__":
    main()