# generate_tsne_proof.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import yaml

from modeling.model_defs import get_patch_encoder
from page.segment_lines import segment_lines
from preprocessor.pipeline import preprocess_line, extract_patches_content_only

# Pick 5 writers from your test set to visualize
TARGET_WRITERS = ["writer_001", "writer_002", "writer_003", "writer_004", "writer_005"]
COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231']

def main():
    cfg = yaml.safe_load(open("config.yml", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    patch = cfg["preprocessing"]["patch_size"]
    emb = cfg["training"]["embedding_dim"]
    model = get_patch_encoder(input_shape=(patch, patch, 1), embedding_dim=emb).to(device)
    checkpoint = torch.load(cfg["paths"]["base_encoder_weights"], map_location=device)
    # Extract the actual weights from inside the 'patch_encoder_state_dict' key
    state_dict = checkpoint.get("patch_encoder_state_dict", checkpoint) 
    model.load_state_dict(state_dict)
    model.eval()

    all_patch_embs, patch_labels = [], []
    all_line_embs, line_labels = [], []

    page_root = Path(cfg["paths"]["page_test_dir"])

    print("Extracting embeddings for t-SNE...")
    for label_idx, writer_id in enumerate(TARGET_WRITERS):
        writer_dir = page_root / writer_id
        if not writer_dir.exists(): continue
            
        page_img = list(writer_dir.glob("*.png"))[0] # Just grab one page per writer
        lines = segment_lines(page_img)
        
        for line_img in lines[:10]: # Process 10 lines per writer
            img, dbg = preprocess_line(line_img, target_h=128, max_w=1580, place="center")
            patches = extract_patches_content_only(img, dbg["content_bounds"])
            
            with torch.no_grad():
                X = torch.from_numpy(np.transpose(patches, (0, 3, 1, 2))).to(device)
                patch_E = model(X).cpu().numpy() # Raw patches
                
            line_E = patch_E.mean(axis=0) # Line pooled
            line_E = line_E / (np.linalg.norm(line_E) + 1e-8)
            
            all_patch_embs.extend(patch_E)
            patch_labels.extend([label_idx] * len(patch_E))
            
            all_line_embs.append(line_E)
            line_labels.append(label_idx)

    print("Running t-SNE... (this takes a moment)")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Raw Patches (The Blob)
    patch_2d = tsne.fit_transform(np.array(all_patch_embs))
    for i in range(len(TARGET_WRITERS)):
        idx = np.array(patch_labels) == i
        ax1.scatter(patch_2d[idx, 0], patch_2d[idx, 1], c=COLORS[i], label=TARGET_WRITERS[i], alpha=0.5, s=10)
    ax1.set_title("A: Flat Patches (No Spatial Structure)")
    
    # Plot 2: Line Pooled (Distinct Clusters)
    line_2d = tsne.fit_transform(np.array(all_line_embs))
    for i in range(len(TARGET_WRITERS)):
        idx = np.array(line_labels) == i
        ax2.scatter(line_2d[idx, 0], line_2d[idx, 1], c=COLORS[i], label=TARGET_WRITERS[i], s=50, edgecolors='black')
    ax2.set_title("B: Hierarchical Line Pooling (Discriminative)")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()