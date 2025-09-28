#!/usr/bin/env python3
"""
evaluate.py
-----------
Evaluation on explicit ./Test:
  - Sample N_POS same-writer pairs (positives)
  - Sample N_NEG different-writer pairs (negatives)
  - Embed each line (preprocess -> content patches -> patch encoder -> mean pool)
  - Compute distances and sweep a threshold to report metrics.
  - Pick threshold from the plotted curves by robust polyline intersection:
        1) Precision == Recall  (balanced operating point)
        2) FPR == FNR           (EER)  [reported for reference]
  - Visualize the sweep and mark both intersections (no saving).
"""

import yaml
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from modeling.model_defs import get_patch_encoder
from preprocessor.pipeline import preprocess_line, extract_patches_content_only
from data.dataset import index_writer_images

# ---------------- Configurable numbers ----------------
N_POS = 1000   # number of same-writer pairs to sample
N_NEG = 1000   # number of different-writer pairs to sample
# ------------------------------------------------------


def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def embed_line(img_path, cfg, patch_encoder, device):
    """
    Convert a line image to a single L2-normalized embedding by:
      preprocess -> extract content-aware patches -> patch encoder -> mean-pool -> L2 norm
    """
    img, dbg = preprocess_line(
        img_path,
        target_h=cfg["preprocessing"]["target_height"],
        max_w=cfg["preprocessing"]["max_width"],
        place=cfg["preprocessing"]["place_eval"],
        use_illum=cfg["preprocessing"]["use_illum"],
        use_clahe=cfg["preprocessing"]["use_clahe"]
    )
    patches = extract_patches_content_only(
        img, dbg["content_bounds"],
        patch=cfg["preprocessing"]["patch_size"],
        stride=cfg["preprocessing"]["stride"],
        min_fg=cfg["preprocessing"]["min_foreground"]
    )  # (N, patch, patch, 1)

    # Run the patch encoder (already L2-normalized outputs)
    with torch.no_grad():
        X = torch.from_numpy(np.transpose(patches, (0, 3, 1, 2))).to(device)  # (N,1,H,W)
        E = patch_encoder.eval()(X).cpu().numpy()  # (N, D)

    e = E.mean(axis=0)                             # mean-pool across patches
    e = e / (np.linalg.norm(e) + 1e-8)             # final L2-normalization
    return e


# ---------- Robust polyline intersection helpers ----------
def _first_valid_crossing_x(x, y):
    """
    Given x (ascending) and y, return x* where y crosses 0 between samples using
    linear interpolation. If multiple crossings, pick the first valid one.
    If none, return None.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.sign(y)
    # valid consecutive pairs (not NaN) with opposite signs or touching zero
    valid = ~np.isnan(y)
    idx = np.where(
        valid[:-1] & valid[1:] & (s[:-1] * s[1:] <= 0)
    )[0]
    if idx.size == 0:
        return None
    i = idx[0]
    x0, x1, y0, y1 = x[i], x[i+1], y[i], y[i+1]
    if y1 == y0:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def _nearest_x_to_zero(x, y):
    """
    Return x at minimal |y| among valid (non-NaN) entries.
    Returns None if all y are NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(y)
    if not np.any(valid):
        return None
    k = int(np.argmin(np.abs(y[valid])))
    return float(x[valid][k])


def _robust_intersection_x(x, y1, y2):
    """
    Return x* where y1(x) == y2(x) under piecewise-linear assumption,
    ignoring degenerate endpoints by treating undefined metrics as NaN.
    Prefers the first valid crossing; falls back to nearest |y1 - y2|.
    """
    x = np.asarray(x, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    d = y1 - y2
    # 1) try a real crossing
    x_star = _first_valid_crossing_x(x, d)
    if x_star is not None:
        return x_star
    # 2) fall back to closest point (no sign change sampled)
    x_min = _nearest_x_to_zero(x, d)
    if x_min is not None:
        return x_min
    # 3) last resort: midpoint (shouldn't happen)
    return float(0.5 * (x[0] + x[-1]))


def main():
    # Load config & set seeds
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)
    set_seeds(cfg["runtime"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load explicit Test set
    test_map = index_writer_images(Path(cfg["paths"]["test_dir"]))
    if not test_map:
        raise RuntimeError("No test data found under ./Test")

    # Prepare patch encoder
    patch = cfg["preprocessing"]["patch_size"]
    emb = cfg["training"]["embedding_dim"]
    patch_encoder = get_patch_encoder(
        input_shape=(patch, patch, 1), embedding_dim=emb
    ).to(device)

    # Load weights saved by train.py
    state = torch.load(cfg["paths"]["base_encoder_weights"], map_location=device)
    # Support both raw state_dict and wrapped dict
    state_dict = state.get("patch_encoder_state_dict", state)
    patch_encoder.load_state_dict(state_dict)
    print("Loaded patch encoder weights.")

    writers = list(test_map.keys())
    pairs, labels = [], []

    # --- Sample positives (same writer) ---
    pos_candidates = []
    for w in writers:
        files = test_map[w]
        if len(files) >= 2:
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    pos_candidates.append((files[i], files[j]))
    random.shuffle(pos_candidates)
    pos_final = pos_candidates[:N_POS]
    pairs.extend(pos_final)
    labels.extend([1] * len(pos_final))

    # --- Sample negatives (different writers) ---
    neg_candidates = []
    for i in range(len(writers)):
        for j in range(i + 1, len(writers)):
            for f1 in test_map[writers[i]]:
                for f2 in test_map[writers[j]]:
                    neg_candidates.append((f1, f2))
    random.shuffle(neg_candidates)
    neg_final = neg_candidates[:N_NEG]
    pairs.extend(neg_final)
    labels.extend([0] * len(neg_final))

    labels = np.array(labels, dtype=int)
    print(f"Sampled {len(pairs)} pairs (positives={sum(labels)}, negatives={len(labels)-sum(labels)})")

    # Distance metric
    metric = cfg["evaluation"]["metric"]

    def dist(a, b):
        if metric == "cosine":
            return 1.0 - float(np.dot(a, b))
        else:
            return float(np.linalg.norm(a - b))

    # Compute distances
    print(f"Scoring {len(pairs)} pairs from ./Test ...")
    dists = []
    for p1, p2 in tqdm(pairs):
        e1 = embed_line(p1, cfg, patch_encoder, device)
        e2 = embed_line(p2, cfg, patch_encoder, device)
        dists.append(dist(e1, e2))
    dists = np.array(dists)

    # Threshold grid
    ths = np.arange(
        cfg["evaluation"]["thresholds"]["start"],
        cfg["evaluation"]["thresholds"]["stop"],
        cfg["evaluation"]["thresholds"]["step"],
    )

    # --- Sweep once to get metrics across the grid (for plotting & intersections)
    accs, precs, recs, f1s, fprs, fnrs = [], [], [], [], [], []
    # Keep raw counts to detect degeneracy
    tps, fps, fns, tns = [], [], [], []
    for th in ths:
        preds = (dists < th).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        # precision with zero_division=np.nan so degenerate thresholds become NaN
        prec, rec, f1_val, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=np.nan
        )
        acc = accuracy_score(labels, preds)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1_val)
        fprs.append(fpr)
        fnrs.append(fnr)
        tps.append(tp); fps.append(fp); fns.append(fn); tns.append(tn)

    accs = np.asarray(accs)
    precs = np.asarray(precs)
    recs  = np.asarray(recs)
    f1s   = np.asarray(f1s)
    fprs  = np.asarray(fprs)
    fnrs  = np.asarray(fnrs)
    tps   = np.asarray(tps); fps = np.asarray(fps); fns = np.asarray(fns); tns = np.asarray(tns)

    # Invalidate endpoints where precision/recall are undefined or trivially both zero
    # (no predicted positives or no actual positives)
    invalid_pr = (np.isnan(precs) | np.isnan(recs) |
                  (tps + fps == 0) | (tps + fns == 0))
    precs_masked = np.where(invalid_pr, np.nan, precs)
    recs_masked  = np.where(invalid_pr, np.nan, recs)

    # --- Intersections "from the graph"
    th_pr  = _robust_intersection_x(ths, precs_masked, recs_masked)   # Precision = Recall
    th_eer = _robust_intersection_x(ths, fprs, fnrs)                   # FPR = FNR (EER)

    # Evaluate final metrics at PR=RC threshold (balanced default)
    preds_bal = (dists < th_pr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds_bal).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds_bal, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds_bal)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    try:
        auc = roc_auc_score(labels, 1.0 - dists)  # higher score = more similar
    except Exception:
        auc = float("nan")

    # --- Report
    print("\n--- Intersections from the graph (robust) ---")
    pr_at = float(np.interp(th_pr, ths, np.nan_to_num(precs_masked, nan=0.0)))
    eer_at = float(np.interp(th_eer, ths, np.nan_to_num(fprs, nan=0.0)))
    print(f"Precision = Recall at threshold ≈ {th_pr:.6f} | value ≈ {pr_at:.4f}")
    print(f"EER (FPR = FNR) at threshold ≈ {th_eer:.6f} | error rate ≈ {eer_at:.4f}")

    print("\n--- Evaluation @ PR=RC threshold ---")
    print(f"Chosen threshold: {th_pr:.6f}")
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"FPR (FAR): {fpr:.4f} | FNR (FRR): {fnr:.4f}")

    # --- Plot (no saving) ---
    # AUC is threshold-independent; use as a reference line
    try:
        auc_ref = roc_auc_score(labels, 1.0 - dists)
    except Exception:
        auc_ref = float("nan")

    plt.figure(figsize=(9, 5))
    plt.plot(ths, accs, label="Accuracy")
    plt.plot(ths, precs, label="Precision")
    plt.plot(ths, recs,  label="Recall")
    plt.plot(ths, f1s,   label="F1")
    plt.plot(ths, fprs,  label="FPR")
    plt.plot(ths, fnrs,  label="FNR")
    plt.plot(ths, [auc_ref]*len(ths), linestyle="--", label=f"AUC (ref) = {auc_ref:.3f}")

    # Mark intersections
    plt.axvline(th_pr, linestyle=":", label=f"PR=RC th = {th_pr:.3f}")
    plt.scatter([th_pr], [pr_at], s=50, marker="o")
    plt.axvline(th_eer, linestyle="--", label=f"EER th = {th_eer:.3f}")
    plt.scatter([th_eer], [eer_at], s=50, marker="x")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold sweep: Accuracy, Precision/Recall/F1, FPR, FNR (AUC as ref)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()