#!/usr/bin/env python3
"""
evaluate.py
-----------
Evaluation on explicit ./Test for Line-Level Verification.
Reports: Accuracy, AUC, Precision, Recall, F1, and Inference Latency.
Includes detailed metrics inside the generated plot.
"""

import yaml, random, time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
)

# fvcore for benchmarking
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

from modeling.model_defs import get_patch_encoder
from preprocessor.pipeline import preprocess_line, extract_patches_content_only
from data.dataset import index_writer_images

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
})

# Sampling configuration
N_POS = 500
N_NEG = 500

def set_seeds(seed: int):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def benchmark_latency(patch_encoder, device, cfg):
    """Measures pure GPU inference time for 1 line (8 patches)."""
    print("\nBenchmarking pure Neural Network Inference Latency...")
    patch_encoder.eval()
    K = cfg["training"]["patches_per_line"]
    patch_size = cfg["preprocessing"]["patch_size"]
    dummy_line = torch.randn(K, 1, patch_size, patch_size).to(device)
    
    with torch.no_grad():
        for _ in range(10): _ = patch_encoder(dummy_line) # Warmup + Lazy Init
        
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    with torch.no_grad():
        for _ in range(100):
            starter.record()
            _ = patch_encoder(dummy_line)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))
    
    return np.mean(timings)

def embed_line(img_path, cfg, patch_encoder, device):
    img, dbg = preprocess_line(
        img_path, target_h=cfg["preprocessing"]["target_height"],
        max_w=cfg["preprocessing"]["max_width"],
        place=cfg["preprocessing"]["place_eval"],
        use_illum=cfg["preprocessing"]["use_illum"], use_clahe=cfg["preprocessing"]["use_clahe"]
    )
    patches = extract_patches_content_only(
        img, dbg["content_bounds"], patch=cfg["preprocessing"]["patch_size"],
        stride=cfg["preprocessing"]["stride"], min_fg=cfg["preprocessing"]["min_foreground"]
    )
    with torch.no_grad():
        X = torch.from_numpy(np.transpose(patches, (0, 3, 1, 2))).to(device)
        E = patch_encoder(X).cpu().numpy()

    e = E.mean(axis=0)
    e = e / (np.linalg.norm(e) + 1e-8)
    return e

def polyline_intersection_x(x, y1, y2):
    x, y1, y2 = np.asarray(x, float), np.asarray(y1, float), np.asarray(y2, float)
    d = y1 - y2
    s = np.sign(d)
    valid = ~np.isnan(d)
    idx = np.where(valid[:-1] & valid[1:] & (s[:-1] * s[1:] <= 0))[0]
    if idx.size:
        i = idx[0]
        x0, x1, d0, d1 = x[i], x[i+1], d[i], d[i+1]
        return float(x0 - d0 * (x1 - x0) / (d1 - d0)) if d1 != d0 else float(x0)
    return float(x[np.nanargmin(np.abs(d))])

def main():
    with open("config.yml", "r") as f: cfg = yaml.safe_load(f)
    set_seeds(cfg["runtime"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_map = index_writer_images(Path(cfg["paths"]["test_dir"]))
    patch_size = cfg["preprocessing"]["patch_size"]
    emb_dim = cfg["training"]["embedding_dim"]
    
    patch_encoder = get_patch_encoder(input_shape=(patch_size, patch_size, 1), embedding_dim=emb_dim).to(device)
    state = torch.load(cfg["paths"]["base_encoder_weights"], map_location=device)
    patch_encoder.load_state_dict(state.get("patch_encoder_state_dict", state))
    patch_encoder.eval()

    # Technical Benchmarks with FVCORE Fix
    params_str, macs_str = "N/A", "N/A"
    if FlopCountAnalysis is not None:
        dummy_input = torch.randn(1, 1, patch_size, patch_size).to(device)
        with torch.no_grad():
            _ = patch_encoder(dummy_input) # Force lazy initialization
            
        flops = FlopCountAnalysis(patch_encoder, dummy_input)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        
        total_p = sum(p.numel() for p in patch_encoder.parameters())
        total_f = flops.total()
        
        params_str = f"{total_p / 1e6:.4f} M"
        if total_f >= 1e9:
            macs_str = f"{total_f / 1e9:.2f} G"
        else:
            macs_str = f"{total_f / 1e6:.2f} M"
            
    latency = benchmark_latency(patch_encoder, device, cfg)

    # Sampling
    writers = list(test_map.keys())
    pairs, labels = [], []
    pos_candidates = [(test_map[w][i], test_map[w][j]) for w in writers if len(test_map[w]) >= 2 for i in range(len(test_map[w])) for j in range(i+1, len(test_map[w]))]
    random.shuffle(pos_candidates)
    pos_final = pos_candidates[:N_POS]
    pairs.extend(pos_final); labels.extend([1] * len(pos_final))

    neg_candidates = [(test_map[writers[i]][f1], test_map[writers[j]][f2]) for i in range(len(writers)) for j in range(i+1, len(writers)) for f1 in range(len(test_map[writers[i]])) for f2 in range(len(test_map[writers[j]]))]
    random.shuffle(neg_candidates)
    neg_final = neg_candidates[:N_NEG]
    pairs.extend(neg_final); labels.extend([0] * len(neg_final))
    labels = np.array(labels, dtype=int)

    # Scoring
    metric = cfg["evaluation"]["metric"]
    dists = []
    for p1, p2 in tqdm(pairs, desc="Scoring Test Pairs"):
        e1, e2 = embed_line(p1, cfg, patch_encoder, device), embed_line(p2, cfg, patch_encoder, device)
        dists.append(1.0 - np.dot(e1, e2) if metric == "cosine" else np.linalg.norm(e1 - e2))
    dists = np.array(dists)

    # Sweep
    ths = np.arange(cfg["evaluation"]["thresholds"]["start"], cfg["evaluation"]["thresholds"]["stop"], cfg["evaluation"]["thresholds"]["step"])
    accs, precs, recs, f1s, fprs, fnrs = [], [], [], [], [], []
    for th in ths:
        preds = (dists < th).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=np.nan)
        accs.append(accuracy_score(labels, preds)); precs.append(p); recs.append(r); f1s.append(f)
        fprs.append(fp/(fp+tn)); fnrs.append(fn/(fn+tp))

    # Metrics at PR=RC
    th_pr = polyline_intersection_x(ths, precs, recs)
    preds_bal = (dists < th_pr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds_bal).ravel()
    acc = accuracy_score(labels, preds_bal)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds_bal, average="binary", zero_division=0)
    auc = roc_auc_score(labels, 1.0 - dists)

    # Final SOTA Table Output
    print("\n" + "="*110)
    print(f"{'Total Params':<12} | {'FLOPs':<10} | {'Inference/Ln':<12} | {'Acc.':<8} | {'AUC':<8} | {'Prec.':<8} | {'Recall':<8} | {'F1':<8} | {'Threshold':<10}")
    print("-"*110)
    print(f"{params_str:<12} | {macs_str:<10} | {latency:>8.2f} ms | {acc*100:>6.2f}% | {auc:.4f} | {p:.4f} | {r:.4f} | {f1:.4f} | {th_pr:.6f}")
    print("="*110 + "\n")

    # --- PLOTTING WITH NUMERICAL METRICS ---
    plt.figure(figsize=(10, 6))
    
    # Adding calculated values directly into the legend labels
    plt.plot(ths, accs, label=f"Accuracy ({acc*100:.1f}%)")
    plt.plot(ths, precs, label=f"Precision ({p:.3f})")
    plt.plot(ths, recs, label=f"Recall ({r:.3f})")
    plt.plot(ths, f1s, label=f"F1 Score ({f1:.3f})")
    plt.plot(ths, fprs, label="FPR")
    plt.plot(ths, fnrs, label="FNR")
    
    # Adding intersection threshold and AUC to the plot
    plt.axvline(th_pr, linestyle=":", color='red', label=f"PR=RC Thres ({th_pr:.3f})")
    plt.plot([], [], ' ', label=f"AUC: {auc:.4f}") # Invisible line just to add AUC to legend
    
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Line-Level Threshold Sweep (Acc: {acc*100:.1f}%, AUC: {auc:.4f})")
    
    # FIX: Restored standard legend positioning so it stays inside the grid
    plt.legend(loc='best') 
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": main()