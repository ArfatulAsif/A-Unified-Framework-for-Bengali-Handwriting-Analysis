# /page/show_lines_segmentation.py


# Just visualize (no saving)
# python -m page.show_lines_segmentation  ./data/Evaluate_For_Pages/218/218_2.jpg

# python -m page.show_lines_segmentation  ./data/Evaluate_For_Pages/216/216_1.jpg


# python -m page.show_lines_segmentation  ./data/Multi_Writer/Test_pages/0001_0002_0003_0006.jpg

# Save crops to a folder and visualize
# python -m page.show_lines_segmentation /path/to/page.jpg --save-dir /tmp/lines


from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Use the exact same logic as your pipeline
from page.segment_lines import segment_lines, detect_line_boxes


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def visualize_segmentation(
    page_img_path: str | Path,
    save_dir: str | Path | None = None,
    margin: int = 4,
    min_words_per_line: int = 1,
    max_cols: int = 3,
):
    """
    Visualize page line segmentation using the SAME segment_lines() used by the pipeline.

    Args:
        page_img_path: path to page image
        save_dir: if provided, segment_lines() will save crops there (and return paths).
                  If None, crops are kept in-memory and just displayed.
        margin: padding around each crop (passed to segment_lines)
        min_words_per_line: drop tiny lines if desired (passed to segment_lines)
        max_cols: max columns for the cropped-line gallery
    """
    page_img_path = Path(page_img_path)
    if not page_img_path.exists():
        raise FileNotFoundError(f"Cannot find image: {page_img_path}")

    # Read original image for overlay
    page_bgr = cv2.imread(str(page_img_path))
    if page_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {page_img_path}")

    # 1) Get crops using the exact pipeline function
    crops: List[Union[np.ndarray, str]] = segment_lines(
        page_img_path=page_img_path,
        output_dir=save_dir,
        margin=margin,
        min_words_per_line=min_words_per_line,
    )

    # 2) Draw boxes for context (same detection logic)
    boxes = detect_line_boxes(page_bgr)

    vis = page_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ---- Figure 1: full page with boxes ----
    plt.figure(figsize=(12, 12))
    plt.imshow(_to_rgb(vis))
    plt.title(f"Line Segmentation — {page_img_path.name}  ({len(crops)} lines)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ---- Figure 2: gallery of cropped lines ----
    if len(crops) == 0:
        print("No lines found.")
        return

    # Load crops into memory if paths were returned
    crop_images: List[np.ndarray] = []
    for c in crops:
        if isinstance(c, (str, Path)):
            img = cv2.imread(str(c))
            if img is None:
                continue
            crop_images.append(img)
        else:
            crop_images.append(c)

    n = len(crop_images)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < n:
                ax.imshow(_to_rgb(crop_images[idx]))
                ax.set_title(f"Line {idx + 1}")
                ax.axis("off")
                idx += 1
            else:
                ax.axis("off")

    fig.suptitle("Segmented Line Crops (from segment_lines())", fontsize=16)
    plt.tight_layout()
    plt.show()

    if save_dir:
        print(f"\nSaved {len(crop_images)} line crops to: {Path(save_dir).resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Visualize page line segmentation.")
    parser.add_argument("page_image", help="Path to a page image")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If provided, save crops here (segment_lines will return file paths).",
    )
    parser.add_argument("--margin", type=int, default=4, help="Crop margin (pixels)")
    parser.add_argument(
        "--min-words-per-line",
        type=int,
        default=1,
        help="Minimum words in a line to keep (filter tiny lines).",
    )
    parser.add_argument(
        "--max-cols", type=int, default=3, help="Max columns for the crop gallery."
    )
    args = parser.parse_args()

    visualize_segmentation(
        page_img_path=args.page_image,
        save_dir=args.save_dir,
        margin=args.margin,
        min_words_per_line=args.min_words_per_line,
        max_cols=args.max_cols,
    )


if __name__ == "__main__":
    main()