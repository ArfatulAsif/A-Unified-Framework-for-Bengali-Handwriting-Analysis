"""
view_preprocessor.py
--------------------
Visualize the *entire* preprocessing evolution for a handwriting line image,
ending with the exact tensor fed to the model.

Pipeline stages shown:
  1) Original (RGB)
  2) Grayscale
  3) Illumination correction (optional)
  4) CLAHE local contrast (optional)
  5) Binary text mask (for cropping)
  6) Tight crop (from source grayscale)
  7) Resized to target height (aspect preserved)
  8) Final canvas (this is the exact model input, normalized to [0,1])

Optional: visualize 128x128 content-aware patches used for patch pooling.




Example:
python -m preprocessor.view_preprocessor --file ./data/Train/2/2_1/2_1_3.jpg --show-patches


python -m preprocessor.view_preprocessor --file ./data/Test/147/147_1/147_1_2.jpg --show-patches


python -m preprocessor.view_preprocessor --file ./data/Test/141/141_2/141_2_2.jpg --show-patches

python -m preprocessor.view_preprocessor --file ./data/Test/141/141_2/141_2_1.jpg --show-patches

python -m preprocessor.view_preprocessor --file ./data/Test/141/141_2/141_2_2.jpg --show-patches



python -m preprocessor.view_preprocessor --file ./data/Train/1/1_1/1_1_1.jpg --show-patches



"""




import argparse
import math
import os
import random
from pathlib import Path
from pathlib import Path
from typing import Optional
import random
import yaml

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import the pipeline and its helpers (underscored funcs are okay to import from our own module)
from .pipeline import (
    preprocess_line,
    extract_patches_content_only,
    _illumination_correct,
    _clahe,
    _adaptive_text_mask,
    _bounds_from_mask,
    _place_on_canvas,
)

# ----------------------------
# Defaults
# ----------------------------





def _find_random_image(root: str) -> str:
    """
    Walk 'root' and return a random image path with common extensions.
    """
    root_path = Path(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    candidates = []
    if root_path.exists():
        for p in root_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                candidates.append(str(p))
    if not candidates:
        raise FileNotFoundError(f"No images found under: {root_path.resolve()}")
    return random.choice(candidates)







with open("config.yml", "r") as f:
    cfg = yaml.safe_load(f)





DEFAULT_ROOT = "./data/Test"   # Fallback root to sample from if --file is not provided
DEFAULT_H = cfg["preprocessing"]["target_height"]
DEFAULT_W = cfg["preprocessing"]["max_width"]
DEFAULT_PATCH = cfg["preprocessing"]["patch_size"]
DEFAULT_STRIDE = cfg["preprocessing"]["stride"]
DEFAULT_MIN_FG = cfg["preprocessing"]["min_foreground"]
DEFAULT_PLACE = cfg["preprocessing"]["place_train"]       # left | center | random
USE_ILLUM = cfg["preprocessing"]["use_illum"] 
USE_CLAHE = cfg["preprocessing"]["use_clahe"] 




def visualize(
    img_path: str,
    target_h: int = cfg["preprocessing"]["target_height"],
    max_w: int = cfg["preprocessing"]["max_width"],
    place: str = cfg["preprocessing"]["place_train"],
    patch: int = cfg["preprocessing"]["patch_size"],
    stride: int = cfg["preprocessing"]["stride"],
    min_fg: float = cfg["preprocessing"]["min_foreground"],
    use_illum: bool = cfg["preprocessing"]["use_illum"],
    use_clahe: bool = cfg["preprocessing"]["use_clahe"],
    show_patches: bool = False,
    save_dir: Optional[str] = None,
    save_final: bool = False,
):
    """
    Run the full preprocessing on a given image and display all key stages.
    Also confirms that the 'final' displayed is exactly what the model gets.
    """
    img_path = str(img_path)
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Recompute the exact stages explicitly (to visualize each one)
    # 1) Grayscale already computed
    # 2) Illumination correction (optional)
    illum = _illumination_correct(gray, ksize=31) if use_illum else gray.copy()

    # 3) CLAHE (optional)
    clahe_img = _clahe(illum, clip=3.0, tiles=8) if use_clahe else illum.copy()

    # 4) Binary text mask + bounds
    mask = _adaptive_text_mask(clahe_img)
    bounds = _bounds_from_mask(mask, smooth=21, col_frac=0.008, row_frac=0.008, margin=4)

    # 5) Tight crop from **source grayscale**
    if bounds is not None:
        y0, y1, x0, x1 = bounds
        cropped_src = gray[y0:y1 + 1, x0:x1 + 1]
    else:
        cropped_src = gray

    # 6) Resize by height, keep aspect
    h, w = cropped_src.shape
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(cropped_src, (new_w, target_h), interpolation=interp)

    # 7) Place on canvas (record content bounds on canvas)
    final_u8, x0_can, x1_can = _place_on_canvas(resized, max_w=max_w, place=place, rng=None)

    # 8) Normalize to [0,1] and add channel -> this is EXACTLY what the model sees
    final = (final_u8.astype("float32") / 255.0)[..., None]

    
    
    # (Optional) Cross-check against pipeline.preprocess_line to confirm identical output
    final_pipe, dbg_pipe = preprocess_line(
        img_path, target_h=target_h, max_w=max_w, place=place,
        use_illum=use_illum, use_clahe=use_clahe
    )

    diff = np.max(np.abs(final - final_pipe))
    same_shape = final.shape == final_pipe.shape

    # ----------------------------
    # Plot all stages
    # ----------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(rgb);                 axes[0, 0].set_title("Original (RGB)");    axes[0, 0].axis("off")
    axes[0, 1].imshow(gray, cmap="gray");   axes[0, 1].set_title("Grayscale");         axes[0, 1].axis("off")
    axes[0, 2].imshow(illum, cmap="gray");  axes[0, 2].set_title("Illumination" if use_illum else "Illumination (skipped)"); axes[0, 2].axis("off")
    axes[0, 3].imshow(clahe_img, cmap="gray"); axes[0, 3].set_title("CLAHE" if use_clahe else "CLAHE (skipped)"); axes[0, 3].axis("off")

    axes[1, 0].imshow(mask, cmap="gray");                 axes[1, 0].set_title("Binary mask");        axes[1, 0].axis("off")
    axes[1, 1].imshow(cropped_src, cmap="gray");          axes[1, 1].set_title("Tight crop (src)");   axes[1, 1].axis("off")
    axes[1, 2].imshow(resized, cmap="gray");              axes[1, 2].set_title(f"Resized (H={target_h})"); axes[1, 2].axis("off")
    im = axes[1, 3].imshow(final.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 3].axvline(x0_can, color="red", linewidth=1)
    axes[1, 3].axvline(x1_can, color="red", linewidth=1)
    axes[1, 3].set_title(f"Final canvas (W={max_w})\n(model input)")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.show()

    # Print a brief summary
    print("\n=== Summary of model input (final) ===")
    print(f"Path: {img_path}")
    print(f"final.shape: {final.shape}, dtype: {final.dtype}")
    print(f"final.min/max: {float(final.min()):.4f} / {float(final.max()):.4f}")
    print(f"Content bounds on canvas (x0,x1): {x0_can}, {x1_can}")
    print(f"Cross-check with preprocess_line -> same_shape: {same_shape}, max_abs_diff: {diff:.8f} [Only take max_abs_diff into consideration if place_train = not random]")

    
    
    
    # Optional: show patches,   # This is my main goal to view
    if show_patches:
        patches = extract_patches_content_only(final, (x0_can, x1_can),
                                               patch=patch, stride=stride, min_fg=min_fg)
        n = patches.shape[0]
        cols = 6
        rows = math.ceil(n / cols)
        plt.figure(figsize=(2.2 * cols, 2.2 * rows))
        for i in range(n):
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(patches[i].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"{i+1}")
            ax.axis("off")
        plt.suptitle(f"Content-aware patches (N={n}) — patch={patch}, stride={stride}, min_fg={min_fg}", fontsize=12)
        plt.tight_layout()
        plt.show()



    # Optional: save outputs
    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem

        cv2.imwrite(str(out_dir / f"{stem}_01_gray.png"), gray)
        cv2.imwrite(str(out_dir / f"{stem}_02_illum.png"), illum)
        cv2.imwrite(str(out_dir / f"{stem}_03_clahe.png"), clahe_img)
        cv2.imwrite(str(out_dir / f"{stem}_04_mask.png"), mask)
        cv2.imwrite(str(out_dir / f"{stem}_05_cropped.png"), cropped_src)
        cv2.imwrite(str(out_dir / f"{stem}_06_resized.png"), resized)
        # Save the exact model input as 8-bit grayscale PNG
        if save_final:
            cv2.imwrite(str(out_dir / f"{stem}_07_final_model_input.png"),
                        (final.squeeze() * 255).astype(np.uint8))
        print(f"Saved intermediates to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Visualize full preprocessing evolution for a handwriting line image.")
    parser.add_argument("--file", "--path", type=str, default=None,
                        help="Path to a line image file. If omitted, a random image is sampled from --root.")
    
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT,
                        help="Root folder to sample from if --file is not provided (e.g., ./data/Test).")
    
    parser.add_argument("--target_h", "--target-h", type=int, default=DEFAULT_H, help="Target line height.")

    parser.add_argument("--max_w", "--max-w", type=int, default=DEFAULT_W, help="Canvas width.")

    parser.add_argument("--place", type=str, default=DEFAULT_PLACE, choices=["left", "center", "random"],
                        help="Where to place the resized line on the canvas.")
    
    parser.add_argument("--patch", type=int, default=DEFAULT_PATCH, help="Patch size for --show-patches.")

    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Patch stride for --show-patches.")

    parser.add_argument("--min_fg", "--min-fg", type=float, default=DEFAULT_MIN_FG,
                        help="Minimum foreground ratio to keep a patch.")


    parser.add_argument("--show_patches", "--show-patches", action="store_true",
                        help="Also visualize content-aware patches.")             # Use this to view patches that goes into the preprocessor
    
    parser.add_argument("--save_dir", "--save-dir", type=str, default=None,
                        help="If set, save intermediate images to this folder.")
    
    parser.add_argument("--save_final", "--save-final", action="store_true",
                        help="If set (and --save_dir given), save the exact model input PNG.")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed when sampling a file from --root.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Choose image
    if args.file:
        img_path = args.file
    else:
        img_path = _find_random_image(args.root)

    print(f"Visualizing: {img_path}")

    visualize(
        img_path=img_path,
        target_h=args.target_h,
        max_w=args.max_w,
        place=args.place,
        patch=args.patch,
        stride=args.stride,
        min_fg=args.min_fg,
        use_illum= USE_ILLUM,
        use_clahe= USE_CLAHE,
        show_patches=args.show_patches,
        save_dir=args.save_dir,
        save_final=args.save_final,
    )


if __name__ == "__main__":
    main()
