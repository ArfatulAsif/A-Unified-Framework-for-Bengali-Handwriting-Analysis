# /page/segment_lines.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Union
import os
import numpy as np
import cv2
import yaml

# ==========================================
# 1. LOAD CONFIGURATION DIRECTLY
# ==========================================
try:
    with open("config.yml", "r") as f:
        _cfg = yaml.safe_load(f)
    _seg_cfg = _cfg.get("segment_lines", {})
except Exception:
    print("[Warning] config.yml not found. Using default segmentation parameters.")
    _seg_cfg = {}

# Set the global defaults directly from config.yml
CFG_MAX_DIM = _seg_cfg.get("max_dim", 1024)
CFG_MARGIN = _seg_cfg.get("margin", 4)
CFG_TOLERANCE = _seg_cfg.get("adaptive_tolerance", 0.5)

# ==========================================
# 2. EASYOCR SETUP
# ==========================================
try:
    import easyocr
except Exception as e:
    raise ImportError(
        "EasyOCR is required for line segmentation. Install with: pip install easyocr\n"
        f"Original error: {e}"
    )

BBox = Tuple[int, int, int, int]  # (x, y, w, h)

# Keep a single Reader instance (GPU if available)
_READER: Optional[easyocr.Reader] = None
def _get_reader() -> easyocr.Reader:
    global _READER
    if _READER is None:
        # Bangla + English
        _READER = easyocr.Reader(['bn', 'en'], gpu=True)
    return _READER


# ---------- Core grouping logic (UPDATED FOR PURE DETECTION) ----------
def _group_detect_boxes_into_lines(h_list: list, tolerance: float = CFG_TOLERANCE) -> List[BBox]:
    if not h_list:
        return []

    def y_center(b):
        return (b[2] + b[3]) / 2.0

    boxes = sorted(h_list, key=y_center)
    lines = []
    current_line_boxes = [boxes[0]]

    for i in range(1, len(boxes)):
        b = boxes[i]
        prev = current_line_boxes[-1]

        b_yc = y_center(b)
        p_yc = y_center(prev)
        p_h  = abs(prev[3] - prev[2]) + 1e-6

        y_tol = tolerance * p_h

        if abs(b_yc - p_yc) < y_tol:
            current_line_boxes.append(b)
        else:
            lines.append(current_line_boxes)
            current_line_boxes = [b]
    lines.append(current_line_boxes)

    line_boxes: List[BBox] = []
    for line in lines:
        x0 = int(min(b[0] for b in line))
        x1 = int(max(b[1] for b in line))
        y0 = int(min(b[2] for b in line))
        y1 = int(max(b[3] for b in line))
        line_boxes.append((x0, y0, x1 - x0, y1 - y0))
        
    return line_boxes


def detect_line_boxes(image_bgr_or_path: Union[str, Path, np.ndarray], max_dim: int = CFG_MAX_DIM) -> List[BBox]:
    if isinstance(image_bgr_or_path, (str, Path)):
        img = cv2.imread(str(image_bgr_or_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_bgr_or_path}")
    else:
        img = image_bgr_or_path
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("image_bgr_or_path must be a file path or a BGR numpy array")

    h, w = img.shape[:2]
    scale = 1.0
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        process_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        process_img = img

    reader = _get_reader()
    horizontal_list, _ = reader.detect(process_img)
    
    if not horizontal_list or not horizontal_list[0]:
        return []
        
    small_boxes = _group_detect_boxes_into_lines(horizontal_list[0])
    
    original_boxes = []
    for (bx, by, bw, bh) in small_boxes:
        ox = int(bx / scale)
        oy = int(by / scale)
        ow = int(bw / scale)
        oh = int(bh / scale)
        original_boxes.append((ox, oy, ow, oh))
        
    return original_boxes


def segment_lines(page_img_path: Union[str, Path],
                  output_dir: Optional[Union[str, Path]] = None,
                  margin: int = CFG_MARGIN,
                  min_words_per_line: int = 1) -> List[Union[np.ndarray, str]]:
    page_img_path = Path(page_img_path)
    img = cv2.imread(str(page_img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {page_img_path}")

    boxes = detect_line_boxes(img)
    H, W = img.shape[:2]
    outputs: List[Union[np.ndarray, str]] = []

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, (x, y, w, h) in enumerate(boxes):
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(W, x + w + margin)
        y1 = min(H, y + h + margin)
        crop = img[y0:y1, x0:x1]

        if output_dir is None:
            outputs.append(crop)
        else:
            out_path = Path(output_dir) / f"line_{i:04d}.png"
            cv2.imwrite(str(out_path), crop)
            outputs.append(str(out_path))

    return outputs