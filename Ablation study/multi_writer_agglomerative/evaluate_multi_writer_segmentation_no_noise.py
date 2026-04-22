# multi_writer/evaluate_multi_writer_segmentation_no_noise.py
# python -m multi_writer_agglomerative.evaluate_multi_writer_segmentation_no_noise

import os
import sys
import yaml
import torch
import cv2
import random
import tempfile
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import AgglomerativeClustering

# Append project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from modeling.model_defs import get_patch_encoder
from preprocessor.pipeline import preprocess_line, extract_patches_content_only
from multi_writer_agglomerative.segment_lines import detect_line_boxes, segment_lines

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

def smooth_sequence(seq, window=3):
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
        if item not in pred_unique: 
            pred_unique.append(item)
            
    for i, cluster_id in enumerate(pred_unique):
        if i < len(gt_unique):
            mapping[cluster_id] = gt_unique[i]
        else:
            mapping[cluster_id] = f"Hallucinated_Writer_{cluster_id}"
            
    mapped_pred = [mapping.get(p, "Unknown") for p in compressed_pred]
    return mapped_pred, mapping

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
    
    # Clustering Params
    distance_threshold = cfg.get("multi_writer", {}).get("distance_threshold", 0.15)
    window_size = cfg.get("multi_writer", {}).get("window_size", 3)
    
    # Load Model
    patch_encoder = get_patch_encoder(
        input_shape=(cfg["preprocessing"]["patch_size"], cfg["preprocessing"]["patch_size"], 1),
        embedding_dim=cfg["training"]["embedding_dim"]
    )
    patch_encoder.load_state_dict(torch.load(cfg["paths"]["base_encoder_weights"], map_location=device)['patch_encoder_state_dict'])
    patch_encoder.to(device).eval()

    # Directories
    test_dir = Path(cfg["paths"]["multi_test_pages"])
    out_dir = Path("./multi_writer/evaluation_output/")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_files = [f for f in test_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".tif", ".png"]]
    
    print(f"\n--- Starting Evaluation on {len(test_files)} pages (Agglomerative) ---")
    
    total_ser = 0.0
    perfect_matches = 0
    color_palette = {} 

    with open(out_dir / "evaluation_report.txt", "w") as log:
        for test_file in test_files:
            page_name = test_file.name
            
            # 1. Parse Ground Truth from filename
            gt_sequence = test_file.stem.split("_")
            if "GT" in gt_sequence: gt_sequence.remove("GT")
                
            # 2. Extract Lines and Embeddings
            boxes = detect_line_boxes(test_file)
            tmp_dir = Path(tempfile.mkdtemp())
            line_paths = segment_lines(test_file, output_dir=tmp_dir)
            
            if not line_paths:
                print(f"[Warning] No lines found in {page_name}. Skipping.")
                continue

            page_embeddings = []
            for path in line_paths:
                e = get_line_embedding(str(path), patch_encoder, cfg, device)
                page_embeddings.append(e)
            
            # 3. Local Agglomerative Clustering
            E_matrix = np.vstack(page_embeddings)
            
            if len(E_matrix) < 2:
                raw_clusters = [0] * len(E_matrix)
            else:
                clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
                raw_clusters = clusterer.fit_predict(E_matrix).tolist()
            
            # 4. Process Sequence and Map to Ground Truth
            smoothed_clusters = smooth_sequence(raw_clusters, window=window_size)
            compressed_clusters = compress_sequence(smoothed_clusters) 
            
            mapped_sequence, mapping_dict = map_clusters_to_gt(compressed_clusters, gt_sequence)
            
            # 5. Calculate Metrics
            edit_dist = levenshtein_distance(gt_sequence, mapped_sequence)
            ser = edit_dist / len(gt_sequence) if len(gt_sequence) > 0 else 0
            
            total_ser += ser
            if edit_dist == 0: perfect_matches += 1

            log_str = f"File: {page_name} | GT: {'_'.join(gt_sequence)} | Pred: {'_'.join(mapped_sequence)} | Edit Dist: {edit_dist} | SER: {ser:.4f}\n"
            print(log_str.strip())
            log.write(log_str)
            
            # 6. Visualization
            img = cv2.imread(str(test_file))
            
            line_writer_ids = [mapping_dict.get(c, "Unknown") for c in raw_clusters]
            
            for (x, y, w, h), wid in zip(boxes, line_writer_ids):
                if wid not in color_palette:
                    color_palette[wid] = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
                color = color_palette[wid]
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                cv2.putText(img, wid, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            cv2.imwrite(str(out_dir / page_name), img)

        # 7. Final Summary
        avg_ser = total_ser / len(test_files) if test_files else 0
        acc = (perfect_matches / len(test_files)) * 100 if test_files else 0
        
        final_summary = f"\n=== FINAL REPORT ===\nAverage SER: {avg_ser:.4f}\nAbsolute Sequence Accuracy: {acc:.2f}%\n"
        print(final_summary)
        log.write(final_summary)

    print("\n--- Writer Color Assignments ---")
    if len(color_palette) > 0:
        legend_h = max(300, len(color_palette) * 40 + 50)
        legend = np.full((legend_h, 400, 3), 255, dtype=np.uint8)
        y_offset = 40
        for wid, color in color_palette.items():
            cv2.rectangle(legend, (20, y_offset-15), (50, y_offset+5), color, -1)
            cv2.putText(legend, f"Writer {wid}", (60, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            y_offset += 40
        
        legend_path = out_dir / "00_color_legend.png"
        cv2.imwrite(str(legend_path), legend)
        print(f"Saved master color legend to: {legend_path}")

if __name__ == "__main__":
    main()