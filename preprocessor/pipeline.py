# pipeline.py
# -----------
# Professional-grade preprocessing for handwriting *line* images:
# 1) Grayscale
# 2) Illumination correction (optional)
# 3) CLAHE local contrast (optional)
# 4) Adaptive text mask + projection-based tight crop
# 5) Resize to target height (keep aspect ratio)
# 6) Place on fixed-width canvas (left/center/random)
# 7) Extract content-aware 128x128 patches for patch-pooling (optional helper)
#
# All steps are designed to preserve stroke geometry and remove shortcuts
# (line length, big margins, uneven background).

from pathlib import Path
import numpy as np
import cv2
import yaml

# --------------------- Core helpers ---------------------

def _illumination_correct(gray, ksize=31):
    """Flatten background illumination by dividing the image by a heavy median blur."""
    ksize = max(3, int(ksize) | 1)  # ensure odd
    bg = cv2.medianBlur(gray, ksize)
    corrected = cv2.divide(gray, bg, scale=255)
    return corrected

def _clahe(gray, clip=3.0, tiles=8):
    """Local contrast enhancement to improve ink visibility and robustness."""
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    return clahe.apply(gray)

def _adaptive_text_mask(gray_eq):
    """
    Build a robust text mask where text=255, background=0.
    Adaptive thresholding + light morphology + CC cleanup.
    """
    block = 35 if min(gray_eq.shape) >= 50 else max(3, (min(gray_eq.shape)//2)*2+1)
    bin_img = cv2.adaptiveThreshold(
        gray_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block, 15
    )
    mask = 255 - bin_img  # invert: text white
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)), iterations=1)

    # Remove tiny components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    H, W = mask.shape
    min_area = max(20, int(0.0005 * H * W))  # ~0.05%
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned

def _bounds_from_mask(mask, smooth=21, col_frac=0.008, row_frac=0.008, margin=4):
    """Convert a binary mask to tight content bounds using smoothed projections."""
    H, W = mask.shape
    if H == 0 or W == 0:
        return None

    col_sum = (mask.sum(axis=0) / 255.0)  # length W
    row_sum = (mask.sum(axis=1) / 255.0)  # length H

    k = max(1, smooth)
    box = np.ones(k, dtype=np.float32) / k
    col_s = np.convolve(col_sum, box, mode="same")
    row_s = np.convolve(row_sum, box, mode="same")

    col_th = max(2.0, col_frac * H)
    row_th = max(2.0, row_frac * W)

    xs = np.where(col_s > col_th)[0]
    ys = np.where(row_s > row_th)[0]
    if xs.size == 0 or ys.size == 0:
        return None

    x0, x1 = int(xs[0]), int(xs[-1])
    y0, y1 = int(ys[0]), int(ys[-1])

    x0 = max(0, x0 - margin); x1 = min(W - 1, x1 + margin)
    y0 = max(0, y0 - margin); y1 = min(H - 1, y1 + margin)
    return (y0, y1, x0, x1)

def _place_on_canvas(resized, max_w, place="left", rng=None):
    """
    Place a resized line (H,new_w) on a fixed-width white canvas.
    Returns: (canvas_uint8, x0_content, x1_content)
    """
    H, new_w = resized.shape
    max_w = int(max_w)
    canvas = np.full((H, max_w), 255, np.uint8)

    if new_w >= max_w:
        canvas[:, :max_w] = resized[:, :max_w]
        return canvas, 0, max_w

    slack = max_w - new_w
    if place == "center":
        left = slack // 2
    elif place == "random":
        if rng is None:
            rng = np.random
        left = int(rng.randint(0, slack + 1))
    else:
        left = 0

    canvas[:, left:left + new_w] = resized
    return canvas, left, left + new_w





# --------------------- Public API ---------------------

with open("config.yml", "r") as f:
    cfg = yaml.safe_load(f)



def preprocess_line(
    img_or_path,
    target_h=cfg["preprocessing"]["target_height"],
    max_w= cfg["preprocessing"]["max_width"],
    place=cfg["preprocessing"]["place_train"],
    use_illum=cfg["preprocessing"]["use_illum"],
    use_clahe= cfg["preprocessing"]["use_clahe"],
    rng=None
):
    """
    Preprocess a line image and return a normalized float tensor plus debug info.

    Returns:
        final_f: float32 array (H, W, 1) in [0,1].
        debug:   dict with intermediates and content bounds (x0, x1).
    """
    # 1) Read grayscale
    if isinstance(img_or_path, (str, Path)):
        gray = cv2.imread(str(img_or_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {img_or_path}")
    else:
        gray = img_or_path
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # 2) Optional illumination correction
    proc = _illumination_correct(gray, ksize=31) if use_illum else gray.copy()

    # 3) Optional local contrast enhancement
    proc_eq = _clahe(proc, clip=3.0, tiles=8) if use_clahe else proc.copy()

    # 4) Build adaptive mask and compute tight content bounds
    mask = _adaptive_text_mask(proc_eq)
    bounds = _bounds_from_mask(mask, smooth=21, col_frac=0.008, row_frac=0.008, margin=4)

    # 5) Crop source grayscale by bounds (fallback to full image if none)
    if bounds is not None:
        y0, y1, x0, x1 = bounds
        cropped_src = gray[y0:y1 + 1, x0:x1 + 1]
    else:
        cropped_src = gray

    # 6) Resize by height (keep aspect ratio)
    h, w = cropped_src.shape
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(cropped_src, (new_w, target_h), interpolation=interp)

    # 7) Place on canvas (record content bounds)
    final_u8, x0_can, x1_can = _place_on_canvas(resized, max_w=max_w, place=place, rng=rng)

    # 8) Normalize to [0,1] and add channel dim (robustly)
    final_f = final_u8.astype("float32") / 255.0
    if final_f.ndim == 2:
        final_f = final_f[..., None]          # (H,W,1) — ensure channel present
    elif final_f.ndim == 3 and final_f.shape[-1] != 1:
        final_f = final_f[..., :1]            # force single channel if needed

    # Extra stats (approximate foreground ratio on final canvas)
    thr = cv2.threshold(final_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    fg_ratio = 1.0 - (thr.mean() / 255.0)

    debug = {
        "final_uint8": final_u8,
        "content_bounds": (x0_can, x1_can),
        "foreground_ratio": fg_ratio,
        "cropped_src_gray": cropped_src,
        "mask": mask,
    }
    return final_f, debug

def extract_patches_content_only(img_hw1, content_bounds, patch=cfg["preprocessing"]["patch_size"], stride=cfg["preprocessing"]["stride"],
                                 min_fg=cfg["preprocessing"]["min_foreground"], max_patches=None):
    """
    Extract sliding patches along the content region only, dropping nearly-empty patches.

    Args:
        img_hw1: float32 image (H, W, 1) in [0,1]
        content_bounds: (x0, x1) from preprocess_line
    Returns:
        np.ndarray of shape (N, patch, patch, 1) — channel axis ALWAYS present
    """
    H, W = img_hw1.shape[:2]
    x0, x1 = content_bounds
    x0 = max(0, int(x0))
    x1 = min(W, int(x1))

    out = []
    x = max(0, x0)
    while True:
        p = img_hw1[:, x:x + patch, :]        # (H, w, 1) or (H, patch, 1)
        if p.shape[1] < patch:
            # Right-pad with white if reaching image end. Keep channel axis.
            p2d = p.squeeze()                 # (H, w)
            p2d = cv2.copyMakeBorder(p2d, 0, 0, 0, patch - p.shape[1],
                                     cv2.BORDER_CONSTANT, value=1.0)
            p = p2d[..., None]                # -> (H, patch, 1)

        # Foreground check on a 2D view
        pu8 = (p.squeeze() * 255).astype(np.uint8)
        thr = cv2.threshold(pu8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        fg = 1.0 - (thr.mean() / 255.0)

        if fg >= float(min_fg):
            # SAFETY: ensure channel axis exists
            if p.ndim == 2:
                p = p[..., None]
            out.append(p.astype(np.float32))

        x += stride
        if x + patch > x1:
            break
        if max_patches is not None and len(out) >= max_patches:
            break

    if not out:
        # Fallback: one centered patch inside content
        xc = max(x0, (x0 + x1 - patch) // 2)
        xc = max(0, min(xc, W - patch))
        p = img_hw1[:, xc:xc + patch, :]
        if p.shape[1] < patch:
            p2d = p.squeeze()
            p2d = cv2.copyMakeBorder(p2d, 0, 0, 0, patch - p.shape[1],
                                     cv2.BORDER_CONSTANT, value=1.0)
            p = p2d[..., None]
        out = [p.astype(np.float32)]

    P = np.stack(out, axis=0)                 # (N, patch, patch, 1) if all elements have channel
    if P.ndim == 3:                           # just-in-case guard → add channel
        P = P[..., None]
    return P