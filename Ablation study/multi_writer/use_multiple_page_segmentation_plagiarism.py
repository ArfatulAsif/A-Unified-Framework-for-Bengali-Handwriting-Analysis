# multi_writer/use_multiple_page_segmentation_plagiarism.py
import os
import sys
import yaml
import torch
import cv2
import random
import tempfile
import argparse
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from modeling.model_defs import get_patch_encoder
from preprocessor.pipeline import preprocess_line, extract_patches_content_only
from multi_writer.segment_lines import detect_line_boxes, segment_lines

def compress_sequence(seq):
    if not seq: return []
    comp = [seq[0]]
    for x in seq[1:]:
        if x != comp[-1]: comp.append(x)
    return comp

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
    parser = argparse.ArgumentParser(description="Scan multiple pages for plagiarism (writer shifts).")
    parser.add_argument("assignment_dir", type=str, help="Path to directory containing assignment pages.")
    args = parser.parse_args()

    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = cfg["evaluation"]["default_threshold"]
    
    patch_encoder = get_patch_encoder(
        input_shape=(cfg["preprocessing"]["patch_size"], cfg["preprocessing"]["patch_size"], 1),
        embedding_dim=cfg["training"]["embedding_dim"]
    )
    patch_encoder.load_state_dict(torch.load(cfg["paths"]["base_encoder_weights"], map_location=device)['patch_encoder_state_dict'])
    patch_encoder.to(device).eval()

    out_dir = Path("./multi_writer/output_for_plagiarism/")
    out_dir.mkdir(parents=True, exist_ok=True)
    assignment_dir = Path(args.assignment_dir)

    writer_centroids = {}
    writer_colors = {}
    writer_count = 0
    multi_writer_flag = False

    page_files = sorted([f for f in assignment_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".tif", ".png"]])
    
    print("\nScanning Assignment Pages...\n")

    for page_path in page_files:
        boxes = detect_line_boxes(page_path)
        tmp_dir = Path(tempfile.mkdtemp())
        line_paths = segment_lines(page_path, output_dir=tmp_dir)

        raw_preds = []
        for path in line_paths:
            e = get_line_embedding(str(path), patch_encoder, cfg, device)
            
            best_id = None
            min_dist = float('inf')
            
            for wid, cent in writer_centroids.items():
                dist = 1.0 - np.dot(e, cent)
                print(dist)
                if dist < min_dist:
                    min_dist = dist
                    best_id = wid
                    
            if best_id is not None and min_dist < threshold:
                raw_preds.append(best_id)
                # Stabilize centroid
                writer_centroids[best_id] = (writer_centroids[best_id] + e) / 2.0
                writer_centroids[best_id] /= np.linalg.norm(writer_centroids[best_id]) + 1e-8
            else:
                writer_count += 1
                new_id = f"writer_{writer_count}"
                writer_centroids[new_id] = e
                writer_colors[new_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                raw_preds.append(new_id)

        comp_preds = compress_sequence(raw_preds)
        print(f"{page_path.name} : {', '.join(comp_preds)}")

        if len(comp_preds) > 1 or writer_count > 1:
            multi_writer_flag = True

        # Visualization
        img = cv2.imread(str(page_path))
        for (x, y, w, h), wid in zip(boxes, raw_preds):
            color = writer_colors[wid]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img, wid, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(out_dir / f"flagged_{page_path.name}"), img)

    if multi_writer_flag:
        print("\n[WARNING] multiple writer found . Plagiarism Flag Triggered!")

if __name__ == "__main__":
    main()