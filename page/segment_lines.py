# /page/segment_lines.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Union
import os
import numpy as np
import cv2

# ---- EasyOCR (with a friendly error if missing) ----
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


# ---------- Core grouping logic (shared by visualizer & embedder) ----------
def _group_words_into_lines(ocr_results: list) -> List[list]:
    """
    Groups EasyOCR word-level results into lines using adaptive tolerance.
    ocr_results: list of [box, text, conf], where box is 4 points (tl,tr,br,bl)
    Returns: list of lines, each = list of word results.
    """
    if not ocr_results:
        return []

    # Sort words top-to-bottom by vertical center
    def y_center(w):
        box = w[0]
        return (box[0][1] + box[2][1]) / 2.0

    words = sorted(ocr_results, key=y_center)

    lines: List[list] = []
    current = [words[0]]

    for i in range(1, len(words)):
        w = words[i]
        prev = current[-1]

        # centers and heights
        w_yc = y_center(w)
        p_yc = y_center(prev)
        p_h  = abs(prev[0][2][1] - prev[0][0][1]) + 1e-6

        # Adaptive tolerance: half of previous word's height (works well in mixed scans)
        y_tol = 0.5 * p_h

        if abs(w_yc - p_yc) < y_tol:
            current.append(w)
        else:
            # start a new line
            lines.append(current)
            current = [w]
    lines.append(current)

    # Within each line, sort left-to-right
    def x_left(w):
        return min(p[0] for p in w[0])
    for line in lines:
        line.sort(key=x_left)

    return lines


def _line_boxes_from_groups(lines: List[list]) -> List[BBox]:
    """Compute (x,y,w,h) bounding boxes per grouped line."""
    boxes: List[BBox] = []
    for line in lines:
        pts = [pt for w in line for pt in w[0]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    # top -> bottom
    boxes.sort(key=lambda b: b[1])
    return boxes


def detect_line_boxes(image_bgr_or_path: Union[str, Path, np.ndarray]) -> List[BBox]:
    """
    Detect line bounding boxes using EasyOCR word boxes + adaptive grouping.
    Returns list of (x,y,w,h) sorted top->bottom.
    """
    if isinstance(image_bgr_or_path, (str, Path)):
        img = cv2.imread(str(image_bgr_or_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_bgr_or_path}")
    else:
        img = image_bgr_or_path
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("image_bgr_or_path must be a file path or a BGR numpy array")

    reader = _get_reader()
    # EasyOCR returns a list of [box, text, conf]
    # Use detail=1 to get boxes; blocklist/allowlist can be tuned later if needed
    results = reader.readtext(img, detail=1, paragraph=False)
    lines = _group_words_into_lines(results)
    boxes = _line_boxes_from_groups(lines)
    return boxes


def segment_lines(page_img_path: Union[str, Path],
                  output_dir: Optional[Union[str, Path]] = None,
                  margin: int = 4,
                  min_words_per_line: int = 1) -> List[Union[np.ndarray, str]]:
    """
    Segment a page into line crops using EasyOCR grouping.
    If output_dir is provided -> saves crops and returns file paths.
    Else -> returns list of BGR numpy arrays.

    Args:
        page_img_path: path to a page image
        output_dir: optional directory to save crops
        margin: pixels of padding added to each crop
        min_words_per_line: drop tiny lines if desired (kept by default)
    """
    page_img_path = Path(page_img_path)
    img = cv2.imread(str(page_img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {page_img_path}")

    # Get raw word detections
    reader = _get_reader()
    word_results = reader.readtext(img, detail=1, paragraph=False)
    # Optionally filter by confidence? (e.g., conf>=0.3) — skip for recall
    # word_results = [w for w in word_results if w[2] >= 0.3]

    # Group and box
    lines = _group_words_into_lines(word_results)
    boxes = _line_boxes_from_groups(lines)

    H, W = img.shape[:2]
    outputs: List[Union[np.ndarray, str]] = []

    # Save or return crops
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, (x, y, w, h) in enumerate(boxes):
        # Optional skip tiny lines
        if min_words_per_line > 1 and len(lines[i]) < min_words_per_line:
            continue

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