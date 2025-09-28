"""
dataset.py
----------
Utility functions for discovering data under ./Train and ./Test.
We assume each dataset directory has the following structure:

    <root>/<writer_id>/<doc_id>/*.jpg

This module returns a dict mapping writer_id -> list of image file paths. Here images are of lines of handwritings. 
"""

import os
from pathlib import Path
from typing import Dict, List
import random

def index_writer_images(root: Path) -> Dict[str, List[str]]:
    """
    Index all images grouped by writer.
    """
    root = Path(root)
    out = {}
    if not root.exists():
        print(f"[WARN] Dataset root not found: {root.resolve()}")
        return out

    print(f"[INFO] Scanning dataset root: {root.resolve()}")

    for writer in sorted(root.iterdir()):
        if not writer.is_dir():
            continue
        files = []
        for doc in sorted(writer.iterdir()):
            if not doc.is_dir():
                continue
            for fn in doc.glob("*.*"):
                if fn.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    files.append(str(fn))
        if files:
            out[writer.name] = sorted(files)
            print(f"  [OK] Writer {writer.name}: {len(files)} images")
        else:
            print(f"  [EMPTY] Writer {writer.name} has no images")

    print(f"[DONE] Found {len(out)} writers under {root}")
    return out


def split_writers(all_writers: Dict[str, List[str]], val_frac: float = 0.2, seed: int = 42):   # slit writers in training and validation set from ./Train
    """
    Split the writers into train and validation sets (writer-disjoint).
    """
    writers = list(all_writers.keys())
    rnd = random.Random(seed)
    rnd.shuffle(writers)

    if len(writers) <= 1:
        return all_writers, {}

    n_val = max(1, int(round(len(writers) * val_frac)))
    val_ids = set(writers[:n_val])

    train = {w: all_writers[w] for w in writers if w not in val_ids}
    val = {w: all_writers[w] for w in writers if w in val_ids}
    return train, val


def sample_two_distinct(items: List[str], rnd: random.Random):
    """
    Sample two distinct items from a list.
    """
    if len(items) < 2:
        raise ValueError("Need at least 2 items to sample two distinct elements.")
    a, b = rnd.sample(items, 2)
    return a, b
