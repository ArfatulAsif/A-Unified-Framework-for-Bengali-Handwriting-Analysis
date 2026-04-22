# multi_writer/tune_multi_writer.py
# python -m multi_writer.tune_multi_writer

import os
import sys
import yaml
import torch
import tempfile
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import DBSCAN

# Append project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from modeling.model_defs import get_patch_encoder
from preprocessor.pipeline import preprocess_line, extract_patches_content_only
from multi_writer.segment_lines import detect_line_boxes, segment_lines

def levenshtein_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]

def smooth_sequence(seq, window=5): # Increased default to 5 for better stability
    if not seq: return []
    pad = window // 2
    smoothed = []
    for i in range(len(seq)):
        start = max(0, i - pad)
        end = min(len(seq), i + pad + 1)
        neighborhood = seq[start:end]
        if not neighborhood: continue
        most_common = Counter(neighborhood).most_common(1)[0][0]
        smoothed.append(most_common)
    return smoothed

def compress_sequence(seq):
    if not seq: return []
    comp = []
    for x in seq:
        if x == -1: # Skip noise entirely
            continue
        if not comp or x != comp[-1]:
            comp.append(x)
    return comp

def map_clusters_to_gt(compressed_pred, gt_sequence):
    mapping = {}
    gt_unique = []
    for item in gt_sequence:
        if item not in gt_unique: gt_unique.append(item)
            
    pred_unique = []
    for item in compressed_pred:
        if item not in pred_unique and item != -1: 
            pred_unique.append(item)
            
    for i, cluster_id in enumerate(pred_unique):
        if i < len(gt_unique):
            mapping[cluster_id] = gt_unique[i]
        else:
            mapping[cluster_id] = f"Hallucinated_Writer_{cluster_id}"
            
    mapping[-1] = "Noise" 
    mapped_pred = [mapping.get(p, "Noise") for p in compressed_pred]
    return mapped_pred

def get_line_embedding(img_path, model, cfg, device):
    img_canvas, dbg = preprocess_line(
        img_path, target_h=cfg["preprocessing"]["target_height"],
        max_w=cfg["preprocessing"]["max_width"], place=cfg["preprocessing"]["place_eval"],
        use_illum=cfg["preprocessing"]["use_illum"], use_clahe=cfg["preprocessing"]["use_clahe"]
    )
    patches = extract_patches_content_only(
        img_canvas, dbg["content_bounds"], patch=cfg["preprocessing"]["patch_size"],
        stride=cfg["preprocessing"]["stride"], min_fg=cfg["preprocessing"]["min_foreground"]
    )
    patches_tensor = torch.from_numpy(np.transpose(patches, (0, 3, 1, 2))).to(device)
    with torch.no_grad():
        embeddings = model(patches_tensor)
    line_emb = embeddings.mean(dim=0)
    return torch.nn.functional.normalize(line_emb, p=2, dim=0).cpu().numpy()

def main():
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    patch_encoder = get_patch_encoder(
        input_shape=(cfg["preprocessing"]["patch_size"], cfg["preprocessing"]["patch_size"], 1),
        embedding_dim=cfg["training"]["embedding_dim"]
    )
    patch_encoder.load_state_dict(torch.load(cfg["paths"]["base_encoder_weights"], map_location=device)['patch_encoder_state_dict'])
    patch_encoder.to(device).eval()

    tune_dir = Path(cfg["paths"]["multi_tune_pages"])
    tune_files = [f for f in tune_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".tif", ".png"]]
    
    print(f"\n--- Phase 1: Extracting & Caching Embeddings for {len(tune_files)} Tuning Pages ---")
    
    # Cache to store embeddings so we don't have to re-run the CNN for every parameter check
    page_cache = {}

    for tune_file in tune_files:
        page_name = tune_file.name
        gt_sequence = tune_file.stem.split("_")
        if "GT" in gt_sequence: gt_sequence.remove("GT")
            
        print(f"Processing {page_name}...")
        tmp_dir = Path(tempfile.mkdtemp())
        line_paths = segment_lines(tune_file, output_dir=tmp_dir)
        
        if not line_paths:
            continue

        embs = []
        for path in line_paths:
            e = get_line_embedding(str(path), patch_encoder, cfg, device)
            embs.append(e)
            
        page_cache[page_name] = {
            "matrix": np.vstack(embs),
            "gt": gt_sequence
        }

    print("\n--- Phase 2: Running Grid Search on Cached Data ---")
    
    # Define our Grid
    eps_values = np.arange(0.04, 0.20, 0.01) # Test eps from 0.04 to 0.19
    min_samples_values = [1, 2, 3]           # Test different density requirements
    window_sizes = [3, 5]                    # Test different smoothing windows
    
    best_ser = float('inf')
    best_params = {}
    
    print(f"Testing {len(eps_values) * len(min_samples_values) * len(window_sizes)} parameter combinations...")
    
    for w_size in window_sizes:
        for min_s in min_samples_values:
            for eps in eps_values:
                
                total_ser = 0.0
                valid_pages = 0
                
                for page_name, data in page_cache.items():
                    E_matrix = data["matrix"]
                    gt_sequence = data["gt"]
                    
                    if len(E_matrix) < min_s:
                        raw_clusters = [0] * len(E_matrix)
                    else:
                        clusterer = DBSCAN(eps=eps, min_samples=min_s, metric="cosine")
                        raw_clusters = clusterer.fit_predict(E_matrix).tolist()
                        
                    smoothed_clusters = smooth_sequence(raw_clusters, window=w_size)
                    compressed_clusters = compress_sequence(smoothed_clusters)
                    mapped_sequence = map_clusters_to_gt(compressed_clusters, gt_sequence)
                    
                    edit_dist = levenshtein_distance(gt_sequence, mapped_sequence)
                    ser = edit_dist / len(gt_sequence) if len(gt_sequence) > 0 else 0
                    
                    total_ser += ser
                    valid_pages += 1
                
                avg_ser = total_ser / valid_pages if valid_pages > 0 else float('inf')
                
                # Update best score
                if avg_ser < best_ser:
                    best_ser = avg_ser
                    best_params = {"eps": round(eps, 3), "min_samples": min_s, "window": w_size}

    print("\n=========================================")
    print("🏆 OPTIMAL HYPERPARAMETERS FOUND 🏆")
    print("=========================================")
    print(f"Optimal eps        : {best_params['eps']}")
    print(f"Optimal min_samples: {best_params['min_samples']}")
    print(f"Optimal window size: {best_params['window']}")
    print(f"Achieved Tuning SER: {best_ser:.4f}")
    print("=========================================\n")
    print("Next step: Update your config.yml with these values and run your test script!")

if __name__ == "__main__":
    main()