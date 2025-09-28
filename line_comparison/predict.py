"""
predict.py
----------
CLI utility to decide if two line images are from the same writer.
Uses the trained patch encoder weights and the preprocessing pipeline.
It outputs the cosine distance and compares it to the threshold in config.yml.


Run it like: 


python -m use_model.predict --config config.yml --img_a ./data/Test/141/141_1/141_1_1.jpg --img_b ./data/Test/141/141_2/141_2_2.jpg

python -m use_model.predict --config config.yml --img_a ./data/Test/142/142_1/142_1_1.jpg --img_b ./data/Test/141/141_2/141_2_2.jpg


python -m use_model.predict --config config.yml --img_a ./data/Test/143/143_1/143_1_1.jpg --img_b ./data/Test/147/147_1/147_1_2.jpg


python -m use_model.predict --config config.yml --img_a ./data/Test/147/147_1/147_1_1.jpg --img_b ./data/Test/147/147_2/147_2_2.jpg


python -m use_model.predict --config config.yml --img_a ./data/Test/141/141_1/141_1_1.jpg --img_b ./data/Test/141/141_2/141_2_1.jpg

python -m use_model.predict --config config.yml --img_a ./data/Test/141/141_1/141_1_1.jpg --img_b ./data/Test/141/141_2/141_2_2.jpg


python -m use_model.predict --config config.yml --img_a ./data/Test/142/142_1/142_1_1.jpg --img_b ./data/Test/141/141_2/141_2_2.jpg


"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# --- Project Imports ---
# Add the project root to the path to allow direct imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from modeling.model_defs import get_patch_encoder
from preprocessor.pipeline import preprocess_line, extract_patches_content_only

def get_line_embedding(img_path: str, model: torch.nn.Module, cfg: dict, device: torch.device) -> np.ndarray:
    """
    Processes a single line image to produce a final, pooled line embedding.
    """
    # 1. Preprocess the line image (resize, place on canvas, etc.)
    img_canvas, dbg = preprocess_line(
        img_path,
        target_h=cfg["preprocessing"]["target_height"],
        max_w=cfg["preprocessing"]["max_width"],
        place=cfg["preprocessing"]["place_eval"], # Use deterministic placement for evaluation
        use_illum=cfg["preprocessing"]["use_illum"],
        use_clahe=cfg["preprocessing"]["use_clahe"]
    )

    # 2. Extract content-aware patches from the line
    patches = extract_patches_content_only(
        img_canvas,
        dbg["content_bounds"],
        patch=cfg["preprocessing"]["patch_size"],
        stride=cfg["preprocessing"]["stride"],
        min_fg=cfg["preprocessing"]["min_foreground"]
    )

    # 3. Convert to PyTorch tensor and move to the correct device
    # Reshape from (N, H, W, C) -> (N, C, H, W) for PyTorch Conv2D
    patches_tensor = torch.from_numpy(np.transpose(patches, (0, 3, 1, 2))).to(device)

    # 4. Get patch embeddings from the model
    # torch.no_grad() disables gradient calculation for faster inference
    with torch.no_grad():
        patch_embeddings = model(patches_tensor) # Shape: (N, embedding_dim)
    
    # 5. Mean pool the patch embeddings to get a single line embedding
    line_embedding = patch_embeddings.mean(dim=0)

    # 6. L2-normalize the final embedding and return as a NumPy array
    line_embedding_normalized = torch.nn.functional.normalize(line_embedding, p=2, dim=0)
    
    return line_embedding_normalized.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Predict if two handwriting images are from the same writer.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yml file.")
    parser.add_argument("--img_a", type=str, required=True, help="Path to the first image (anchor).")
    parser.add_argument("--img_b", type=str, required=True, help="Path to the second image to compare.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional: Override the default decision threshold.")
    args = parser.parse_args()

    # --- Load Config ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print("Loading trained patch encoder model...")
    patch_encoder = get_patch_encoder(
        input_shape=(cfg["preprocessing"]["patch_size"], cfg["preprocessing"]["patch_size"], 1),
        embedding_dim=cfg["training"]["embedding_dim"]
    )
    
    weights_path = cfg["paths"]["base_encoder_weights"]
    state_dict = torch.load(weights_path, map_location=device)
    patch_encoder.load_state_dict(state_dict['patch_encoder_state_dict'])
    
    patch_encoder.to(device)
    patch_encoder.eval() # Set model to evaluation mode (disables dropout, etc.)

    # --- Get Embeddings ---
    print(f"\nProcessing Image A: {Path(args.img_a).name}")
    embedding_a = get_line_embedding(args.img_a, patch_encoder, cfg, device)
    
    print(f"Processing Image B: {Path(args.img_b).name}")
    embedding_b = get_line_embedding(args.img_b, patch_encoder, cfg, device)

    # --- Compare and Predict ---
    # For L2-normalized vectors, cosine similarity is the dot product.
    # Cosine Distance = 1 - Cosine Similarity
    cosine_similarity = np.dot(embedding_a, embedding_b)
    cosine_distance = 1 - cosine_similarity

    threshold = args.threshold if args.threshold is not None else cfg["evaluation"]["default_threshold"]
    
    if cosine_distance < threshold:
        decision = "Same Writer"
    else:
        decision = "Different Writers"

    # --- Print Results ---
    print("\n--- Prediction Result ---")
    print(f"Cosine Distance: {cosine_distance:.4f}")
    print(f"Decision Threshold: {threshold:.4f}")
    print(f"Decision: ==> {decision}")


if __name__ == "__main__":
    main()