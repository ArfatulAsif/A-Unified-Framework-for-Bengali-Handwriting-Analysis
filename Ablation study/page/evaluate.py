# /page/evaluate.py
from __future__ import annotations
import yaml, random, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
)

from page.page_embedding import load_patch_encoder, embed_page

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# How many pairs to sample
N_POS = 150
N_NEG = 150

def set_seeds(seed: int):
    import random as pyrand
    np.random.seed(seed); pyrand.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def index_writer_pages(root: Path) -> Dict[str, List[Path]]:
    """Expect: root/<writer_id>/*.(png|jpg|jpeg|tif|bmp)"""

    writers: Dict[str, List[Path]] = {}

    for wdir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            imgs.extend(sorted(wdir.glob(ext)))
        if imgs:
            writers[wdir.name] = imgs
            print(wdir.name)

    return writers

def distance(e1: np.ndarray, e2: np.ndarray, metric: str) -> float:
    return 1.0 - float(np.dot(e1, e2)) if metric == "cosine" else float(np.linalg.norm(e1 - e2))

def polyline_intersection_x(x, y1, y2):
    """
    Piecewise-linear intersection x*(y1==y2). Robust to NaNs.
    Prefer first valid crossing; otherwise nearest |y1-y2|.
    """
    x = np.asarray(x, float)
    d = np.asarray(y1, float) - np.asarray(y2, float)
    valid = ~np.isnan(d)
    xv, dv = x[valid], d[valid]
    if len(xv) < 2:
        return float(x[0])
    s = np.sign(dv)
    idx = np.where(s[:-1] * s[1:] <= 0)[0]
    if idx.size:
        i = idx[0]
        x0, x1, d0, d1 = xv[i], xv[i+1], dv[i], dv[i+1]
        if d1 == d0:
            return float(x0)
        return float(x0 - d0 * (x1 - x0) / (d1 - d0))
    k = int(np.argmin(np.abs(dv)))
    if 0 < k < len(dv) - 1:
        i = k - 1 if abs(dv[k-1]) < abs(dv[k+1]) else k
        x0, x1, d0, d1 = xv[i], xv[i+1], dv[i], dv[i+1]
        if d1 != d0:
            x_star = float(x0 - d0 * (x1 - x0) / (d1 - d0))
            return float(min(max(x_star, x0), x1))
    return float(xv[k])


# ==========================================
# MODEL INFERENCE TIMER WRAPPER
# ==========================================
class InferenceTimerWrapper(nn.Module):
    """
    Wraps the patch encoder to accurately measure pure GPU/CPU inference time,
    completely isolating it from OpenCV preprocessing/segmentation steps.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.total_inference_time = 0.0

    def forward(self, *args, **kwargs):
        # Synchronize before starting the timer to ensure accurate GPU measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        result = self.model(*args, **kwargs)
        
        # Synchronize after the forward pass before stopping the timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total_inference_time += time.perf_counter() - start
        
        return result


def main():
    cfg = yaml.safe_load(open("config.yml", "r"))
    seed = cfg.get("runtime", {}).get("seed", 123)
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    page_root = Path(cfg["paths"]["page_test_dir"])
    metric = cfg["evaluation_page"]["metric"]

    wmap = index_writer_pages(page_root)
    if not wmap:
        raise SystemExit(f"No page data under {page_root}")

    # Load and wrap the model to track pure inference time
    raw_patch_encoder = load_patch_encoder(cfg, device)
    patch_encoder = InferenceTimerWrapper(raw_patch_encoder)
    patch_encoder.eval()

    # ---- sample page pairs ----
    writers = list(wmap.keys())

    pos_candidates: List[Tuple[Path, Path]] = []
    for w in writers:
        files = wmap[w]
        if len(files) >= 2:
            for i in range(len(files)):
                for j in range(i+1, len(files)):
                    pos_candidates.append((files[i], files[j]))
    random.shuffle(pos_candidates)
    pos_pairs = pos_candidates[:N_POS]

    neg_candidates: List[Tuple[Path, Path]] = []
    for i in range(len(writers)):
        for j in range(i+1, len(writers)):
            for f1 in wmap[writers[i]]:
                for f2 in wmap[writers[j]]:
                    neg_candidates.append((f1, f2))
    random.shuffle(neg_candidates)
    neg_pairs = neg_candidates[:N_NEG]

    pairs = pos_pairs + neg_pairs
    labels = np.array([1]*len(pos_pairs) + [0]*len(neg_pairs), dtype=int)

    print(f"Sampled {len(pairs)} page pairs (positives={len(pos_pairs)}, negatives={len(neg_pairs)})")

    # ---- embed & score with progress, caching, and timing ----
    cache: Dict[Path, np.ndarray] = {}
    page_total_times: List[float] = []

    def get_embed(p: Path) -> np.ndarray | None:
        if p in cache:
            return cache[p]
        
        # Time the total end-to-end processing of a single page
        start_time = time.perf_counter()
        emb = embed_page(p, cfg, patch_encoder, device)
        end_time = time.perf_counter()
        
        # Record processing time
        page_total_times.append(end_time - start_time)
        
        cache[p] = emb
        return emb

    dists: List[float] = []
    for p1, p2 in tqdm(pairs, desc="Embedding & scoring pages", unit="pair"):
        e1 = get_embed(p1)
        e2 = get_embed(p2)
        if e1 is None or e2 is None:
            dists.append(2.0)  # max-ish distance if something failed
        else:
            dists.append(distance(e1, e2, metric))
    dists = np.asarray(dists, float)

    # ---- threshold sweep with progress ----
    th_cfg = cfg["evaluation_page"]["thresholds"]
    ths = np.arange(float(th_cfg["start"]), float(th_cfg["stop"]), float(th_cfg["step"]))

    accs, precs, recs, f1s, fprs, fnrs = [], [], [], [], [], []
    for idx in trange(len(ths), desc="Threshold sweep", unit="th"):
        th = ths[idx]
        preds = (dists < th).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=np.nan)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        accs.append(acc); precs.append(prec); recs.append(rec)
        f1s.append(f1); fprs.append(fpr); fnrs.append(fnr)

    accs, precs, recs, f1s, fprs, fnrs = map(np.asarray, (accs, precs, recs, f1s, fprs, fnrs))

    # AUC reference
    try:
        auc_ref = roc_auc_score(labels, 1.0 - dists)
    except Exception:
        auc_ref = float("nan")

    # ---- intersections from curves ----
    th_pr  = polyline_intersection_x(ths, precs, recs)   # Precision = Recall
    th_eer = polyline_intersection_x(ths, fprs, fnrs)    # FPR = FNR

    # ---- metrics @ PR=RC ----
    preds_bal = (dists < th_pr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds_bal).ravel()
    acc = accuracy_score(labels, preds_bal)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds_bal, average="binary", zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print("\n" + "="*50)
    print("--- TIMING BENCHMARKS ---")
    
    # Calculate Averages
    avg_total_time = np.mean(page_total_times) if page_total_times else 0.0
    num_unique_pages = len(cache)
    avg_model_inference = patch_encoder.total_inference_time / max(1, num_unique_pages)
    avg_preprocessing = avg_total_time - avg_model_inference

    print(f"Total Unique Pages Processed : {num_unique_pages}")
    print(f"Avg Total Time Per Page      : {avg_total_time:.4f} sec")
    print(f"  ├─ Avg Preprocessing Time  : {avg_preprocessing:.4f} sec  (Segmentation, Resizing, etc.)")
    print(f"  └─ Avg Model Inference Time: {avg_model_inference:.4f} sec  (Pure Neural Network Forward Pass)")
    print("="*50)

    print("\n--- Page-level intersections ---")
    print(f"PR=RC threshold ≈ {th_pr:.6f} | PR=RC value ≈ {np.interp(th_pr, ths, np.nan_to_num(precs, nan=0.0)):.4f}")
    print(f"EER threshold   ≈ {th_eer:.6f} | error rate ≈ {np.interp(th_eer, ths, np.nan_to_num(fprs, nan=0.0)):.4f}")

    print("\n--- Page-level metrics @ PR=RC ---")
    print(f"Threshold: {th_pr:.6f}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc_ref:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"FPR: {fpr:.4f} | FNR: {fnr:.4f}")

    # ---- plot (no saving) ----
    plt.figure(figsize=(10, 5))
    plt.plot(ths, accs, label="Accuracy")
    plt.plot(ths, precs, label="Precision")
    plt.plot(ths, recs,  label="Recall")
    plt.plot(ths, f1s,   label="F1")
    plt.plot(ths, fprs,  label="FPR")
    plt.plot(ths, fnrs,  label="FNR")
    plt.plot(ths, [auc_ref]*len(ths), linestyle="--", label=f"AUC (ref) = {auc_ref:.3f}")
    plt.axvline(th_pr,  linestyle=":",  label=f"PR=RC th = {th_pr:.3f}")
    plt.axvline(th_eer, linestyle="--", label=f"EER th = {th_eer:.3f}")
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.title("Page-level Threshold Sweep")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()