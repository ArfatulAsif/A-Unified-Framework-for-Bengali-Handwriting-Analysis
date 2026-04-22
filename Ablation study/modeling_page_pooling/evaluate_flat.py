# evaluate_flat_vit.py
# --------------------
# Evaluates the SOTA Transformer [CLS] Token Flat architecture.

from __future__ import annotations
import yaml, random, time, cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from modeling_page_pooling.model_defs_advanced import get_patch_encoder, get_page_encoder

def set_seeds(seed: int):
    import random as pyrand
    np.random.seed(seed); pyrand.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def index_writer_pages(root: Path) -> Dict[str, List[Path]]:
    writers: Dict[str, List[Path]] = {}
    for wdir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            imgs.extend(sorted(wdir.glob(ext)))
        if imgs: writers[wdir.name] = imgs
    return writers

def load_advanced_encoder(cfg, device):
    patch = cfg["preprocessing"]["patch_size"]
    emb = cfg["training"]["embedding_dim"]
    
    patch_encoder = get_patch_encoder(input_shape=(patch, patch, 1), embedding_dim=emb)
    page_encoder = get_page_encoder(patch_encoder, embedding_dim=emb).to(device)
    
    base_path = cfg["paths"]["base_encoder_weights"]
    
    try:
        checkpoint = torch.load(base_path, map_location=device)
        state_dict = checkpoint.get("page_encoder_state_dict", checkpoint)
        page_encoder.load_state_dict(state_dict)
        print(f"[INFO] Successfully loaded weights from {base_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        
    page_encoder.eval()
    return page_encoder

def embed_page_vit(page_img_path: Path, cfg: dict, page_encoder: nn.Module, device: torch.device) -> Tuple[Optional[np.ndarray], float, float]:
    t0_total = time.time()
    
    img = cv2.imread(str(page_img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None, 0.0, 0.0

    K = 32
    patch_size = cfg["preprocessing"].get("patch_size", 128)
    min_fg = cfg["preprocessing"].get("min_foreground", 0.04)
    max_dim = cfg.get("segment_lines", {}).get("max_dim", 1024)

    h, w = img.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if cfg["preprocessing"].get("use_clahe", True):
        img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)

    thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    fg_mask = 1.0 - (thr.astype("float32") / 255.0)
    img_f = img.astype("float32") / 255.0

    H, W = img_f.shape
    valid_patches = []
    stride = cfg["preprocessing"].get("stride", 56)
    
    for y in range(0, max(1, H - patch_size + 1), stride):
        for x in range(0, max(1, W - patch_size + 1), stride):
            y_end, x_end = min(y + patch_size, H), min(x + patch_size, W)
            p_fg = fg_mask[y:y_end, x:x_end]
            if (np.mean(p_fg) if p_fg.size > 0 else 0) >= min_fg:
                patch = img_f[y:y_end, x:x_end]
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    patch = cv2.copyMakeBorder(patch, 0, patch_size - patch.shape[0], 
                                               0, patch_size - patch.shape[1], 
                                               cv2.BORDER_CONSTANT, value=1.0)
                valid_patches.append(patch)

    if not valid_patches: return None, 0.0, 0.0

    if len(valid_patches) >= K:
        rng = np.random.RandomState(sum([ord(c) for c in page_img_path.name]))
        idx = rng.choice(len(valid_patches), K, replace=False)
        sampled_patches = np.stack([valid_patches[i] for i in idx], axis=0)
    else:
        sampled_patches = np.stack([valid_patches[i % len(valid_patches)] for i in range(K)], axis=0)

    sampled_patches = np.expand_dims(sampled_patches, axis=(0, 2)) # (1, K, 1, 128, 128)
    
    # --- Track Inference Time ---
    t0_infer = time.time()
    with torch.no_grad():
        X = torch.from_numpy(sampled_patches).to(device)
        page_emb = page_encoder(X).cpu().numpy()[0] 
    t_infer_total = time.time() - t0_infer
    
    t_total = time.time() - t0_total
        
    return page_emb, t_total, t_infer_total

def main():
    cfg = yaml.safe_load(open("config.yml", "r"))
    set_seeds(cfg.get("runtime", {}).get("seed", 123))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wmap = index_writer_pages(Path(cfg["paths"]["page_test_dir"]))
    page_encoder = load_advanced_encoder(cfg, device)

    N_POS, N_NEG = 150, 150
    writers = list(wmap.keys())
    
    pos_cands, neg_cands = [], []
    for w in writers:
        files = wmap[w]
        for i in range(len(files)):
            for j in range(i+1, len(files)): pos_cands.append((files[i], files[j]))
    random.shuffle(pos_cands)
    pos_pairs = pos_cands[:N_POS]

    for i in range(len(writers)):
        for j in range(i+1, len(writers)):
            for f1 in wmap[writers[i]]:
                for f2 in wmap[writers[j]]: neg_cands.append((f1, f2))
    random.shuffle(neg_cands)
    neg_pairs = neg_cands[:N_NEG]

    pairs = pos_pairs + neg_pairs
    labels = np.array([1]*len(pos_pairs) + [0]*len(neg_pairs), dtype=int)

    cache = {}
    dists = []
    metric = cfg["evaluation_page"]["metric"]
    
    total_processing_time = 0.0
    total_inference_time = 0.0
    pages_processed = 0
    
    for p1, p2 in tqdm(pairs, desc="Evaluating SOTA ViT Model"):
        if p1 not in cache: 
            emb1, t_tot1, t_inf1 = embed_page_vit(p1, cfg, page_encoder, device)
            cache[p1] = emb1
            if emb1 is not None:
                total_processing_time += t_tot1
                total_inference_time += t_inf1
                pages_processed += 1

        if p2 not in cache: 
            emb2, t_tot2, t_inf2 = embed_page_vit(p2, cfg, page_encoder, device)
            cache[p2] = emb2
            if emb2 is not None:
                total_processing_time += t_tot2
                total_inference_time += t_inf2
                pages_processed += 1
                
        e1, e2 = cache[p1], cache[p2]
        
        if e1 is None or e2 is None or np.isnan(e1).any() or np.isnan(e2).any(): 
            dists.append(2.0)
        else: 
            val = 1.0 - float(np.dot(e1, e2)) if metric == "cosine" else float(np.linalg.norm(e1 - e2))
            dists.append(val)
    
    dists = np.asarray(dists, float)

    th_cfg = cfg["evaluation_page"]["thresholds"]
    ths = np.arange(float(th_cfg["start"]), float(th_cfg["stop"]), float(th_cfg["step"]))
    
    precs, recs = [], []
    for th in ths:
        preds = (dists < th).astype(int)
        p, r, _, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        precs.append(p); recs.append(r)

    precs_arr = np.array(precs)
    recs_arr = np.array(recs)
    
    valid_mask = (precs_arr + recs_arr) > 0
    diffs = np.where(valid_mask, np.abs(precs_arr - recs_arr), np.inf)
    
    best_idx = np.argmin(diffs)
    th_pr = ths[best_idx]
    
    preds_bal = (dists < th_pr).astype(int)
    acc = accuracy_score(labels, preds_bal)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds_bal, average="binary", zero_division=0)
    
    # Calculate AUC
    try:
        auc_score = roc_auc_score(labels, 1.0 - dists)
    except:
        auc_score = float('nan')

    print("\n" + "="*50)
    print(" SOTA ViT [CLS] TOKEN PAGE EVALUATION ")
    print("="*50)
    
    if pages_processed > 0:
        avg_tot = total_processing_time / pages_processed
        avg_inf = total_inference_time / pages_processed
        print(f"Avg Total Time per Page     : {avg_tot:.4f} sec")
        print(f"Avg Inference Time per Page : {avg_inf:.4f} sec")
        print(f"Avg Prep/Sampling Time      : {(avg_tot - avg_inf):.4f} sec")
        print("-" * 50)

    print(f"Threshold: {th_pr:.6f}")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"AUC      : {auc_score:.4f}")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()