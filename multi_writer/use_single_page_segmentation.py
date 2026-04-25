# multi_writer/use_single_page_segment.py
# Usage: python -m multi_writer.use_single_page_segmentation --image path/to/your/image.jpg

#  python -m multi_writer.use_single_page_segmentation --image ./data/Multi_Writer/Test_pages/0000_0001_0002_0004.jpg

import os
import sys
import argparse
import yaml
import torch
import cv2
import random
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

def smooth_sequence(seq, window=3):
    """Moving majority to fix single-line OCR/clustering noise."""
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
    """Collapse consecutive duplicates. Noise (-1) is INCLUDED as a distinct segment."""
    if not seq: return []
    comp = []
    for x in seq:
        if not comp or x != comp[-1]:
            comp.append(x)
    return comp

def map_clusters_to_sequential_names(compressed_pred):
    """
    Maps anonymous DBSCAN cluster IDs to sequential names (Writer_1, Writer_2)
    based strictly on their chronological order of appearance on the page.
    """
    mapping = {}
    writer_count = 1
    
    # Extract unique IDs in order of appearance
    pred_unique = []
    for item in compressed_pred:
        if item not in pred_unique and item != -1: 
            pred_unique.append(item)
            
    # Assign sequential names
    for cluster_id in pred_unique:
        mapping[cluster_id] = f"Writer_{writer_count}"
        writer_count += 1
            
    mapping[-1] = "Noise" # Explicitly map -1 to "Noise"

    # Apply mapping back to the compressed sequence
    mapped_pred = [mapping.get(p, "Noise") for p in compressed_pred]
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
    parser = argparse.ArgumentParser(description="Segment a single document by multiple writers.")
    parser.add_argument("--image", type=str, required=True, help="Path to the handwritten document image.")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found at {img_path}")
        sys.exit(1)

    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters pulled dynamically from your tuned config
    eps = cfg.get("multi_writer", {}).get("eps", 0.08)
    min_samples = cfg.get("multi_writer", {}).get("min_samples", 2)
    window_size = cfg.get("multi_writer", {}).get("window_size", 3)
    
    # Load Model
    print("Loading Patch Encoder model...")
    patch_encoder = get_patch_encoder(
        input_shape=(cfg["preprocessing"]["patch_size"], cfg["preprocessing"]["patch_size"], 1),
        embedding_dim=cfg["training"]["embedding_dim"]
    )
    patch_encoder.load_state_dict(torch.load(cfg["paths"]["base_encoder_weights"], map_location=device)['patch_encoder_state_dict'])
    patch_encoder.to(device).eval()

    # Directories
    out_dir = Path("./multi_writer/single_page_output/")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing Document: {img_path.name} ---")
    
    # 1. Extract Lines and Embeddings
    print("Extracting handwriting lines...")
    boxes = detect_line_boxes(img_path)
    tmp_dir = Path(tempfile.mkdtemp())
    line_paths = segment_lines(img_path, output_dir=tmp_dir)
    
    if not line_paths:
        print("[ERROR] No handwriting lines detected in the image.")
        sys.exit(1)

    print(f"Extracted {len(line_paths)} lines. Computing deep embeddings...")
    page_embeddings = []
    for path in line_paths:
        e = get_line_embedding(str(path), patch_encoder, cfg, device)
        page_embeddings.append(e)
    
    # 2. Local DBSCAN Clustering
    print(f"Clustering lines (eps={eps}, min_samples={min_samples}, window={window_size})...")
    E_matrix = np.vstack(page_embeddings)
    
    if len(E_matrix) < min_samples:
        raw_clusters = [0] * len(E_matrix)
    else:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        raw_clusters = clusterer.fit_predict(E_matrix).tolist()
    
    # 3. Process Sequence and Map to Sequential Names
    smoothed_clusters = smooth_sequence(raw_clusters, window=window_size)
    compressed_clusters = compress_sequence(smoothed_clusters) 
    mapped_sequence, mapping_dict = map_clusters_to_sequential_names(compressed_clusters)
    
    # 4. Extract Total Unique Writers (Ignoring Noise)
    total_writers = len([w for w in set(mapped_sequence) if w != "Noise"])
    
    # 5. Visualization
    print("Drawing bounding boxes and saving output...")
    img = cv2.imread(str(img_path))
    
    # Initialize colors (Red for Noise)
    color_palette = {"Noise": (0, 0, 255)} 
    
    # Get the mapped sequential name for every single physical line
    line_writer_labels = [mapping_dict.get(c, "Noise") for c in raw_clusters]
    
    for (x, y, w, h), label in zip(boxes, line_writer_labels):
        if label not in color_palette:
            color_palette[label] = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        color = color_palette[label]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    out_file = out_dir / f"segmented_{img_path.name}"
    cv2.imwrite(str(out_file), img)

    # 6. Final Report
    print("\n=========================================")
    print("📝 INFERENCE REPORT")
    print("=========================================")
    print(f"Total Unique Writers Detected: {total_writers}")
    print(f"Chronological Sequence       : {' -> '.join(mapped_sequence)}")
    print(f"Annotated Image Saved To     : {out_file.resolve()}")
    print("=========================================\n")

if __name__ == "__main__":
    main()