# /page/page_embedding.py
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch

from page.segment_lines import segment_lines
from preprocessor.pipeline import preprocess_line, extract_patches_content_only
from modeling.model_defs import get_patch_encoder

def load_patch_encoder(cfg, device):
    patch = cfg["preprocessing"]["patch_size"]
    emb = cfg["training"]["embedding_dim"]
    model = get_patch_encoder(input_shape=(patch, patch, 1), embedding_dim=emb).to(device)
    state = torch.load(cfg["paths"]["base_encoder_weights"], map_location=device)
    state_dict = state.get("patch_encoder_state_dict", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def embed_line_from_path(line_img_path: str, cfg, patch_encoder, device) -> np.ndarray:
    img, dbg = preprocess_line(
        line_img_path,
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
    )  # (N, H, W, 1)

    with torch.no_grad():
        X = torch.from_numpy(np.transpose(patches, (0, 3, 1, 2))).to(device)  # (N,1,H,W)
        E = patch_encoder(X).cpu().numpy()  # (N, D)

    e = E.mean(axis=0)
    e = e / (np.linalg.norm(e) + 1e-8)
    return e

def embed_page(page_img_path: str | Path, cfg, patch_encoder, device) -> Optional[np.ndarray]:
    """
    Segment page -> embed each line -> mean-pool -> L2 norm.
    Returns None if no lines found.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="page_lines_"))
    try:
        line_paths = segment_lines(page_img_path, output_dir=tmp_dir)
        if len(line_paths) == 0:
            return None
        line_embs: List[np.ndarray] = []
        for p in line_paths:
            e = embed_line_from_path(str(p), cfg, patch_encoder, device)
            if e is not None:
                line_embs.append(e)
        if len(line_embs) == 0:
            return None
        page_emb = np.mean(line_embs, axis=0)
        page_emb = page_emb / (np.linalg.norm(page_emb) + 1e-8)
        return page_emb
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)