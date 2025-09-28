# train.py
# --------
# Training script that:
#   - indexes ./Train
#   - splits writers into train/val (if no explicit val_dir is provided)
#   - builds a patch-encoder + line-encoder + triplet siamese
#   - trains with triplet loss on pooled line embeddings (using K patches per line)
#   - saves best weights for the patch encoder (used by evaluation and prediction)

import yaml, random, os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---- PyTorch imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---- Project modules
# These imports pull in functions from other files in your project.
from data.dataset import index_writer_images, split_writers, sample_two_distinct
from preprocessor.pipeline import preprocess_line, extract_patches_content_only
from modeling.model_defs import get_patch_encoder, get_line_encoder, build_triplet_siamese


def set_seeds(seed: int):
    """Set all RNG seeds for reproducibility."""
    # Ensures that random operations are the same every time you run the script.
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Optional: enable mixed precision on GPU for faster training and less memory usage.
if torch.cuda.is_available():
    os.environ["TORCH_ALLOW_TF32"] = "1"
    # GradScaler helps prevent numerical issues (underflow) when using float16.
    scaler = torch.cuda.amp.GradScaler()
    print("Mixed precision ENABLED:", "mixed_float16")
else:
    scaler = None
    print("Mixed precision DISABLED (no GPU detected).")


class TripletPatchGenerator:
    """
    This class is a custom data generator. Its job is to load image paths,
    preprocess them on-the-fly, and yield batches of (Anchor, Positive, Negative) triplets
    for the model to train on.
    """
    def __init__(self, writers_dict, cfg, is_train=True):
        # A triplet requires an Anchor and a Positive from the same writer, so we
        # filter out any writers who don't have at least 2 images.
        self.writers_dict = {w: v for w, v in writers_dict.items() if len(v) >= 2}
        self.writer_ids = list(self.writers_dict.keys())
        self.cfg = cfg
        self.is_train = is_train

        # Cache config values for quick access during batch generation.
        self.bs = cfg["training"]["batch_size"]
        self.K = cfg["training"]["patches_per_line"]
        self.H = cfg["preprocessing"]["target_height"]
        self.W = cfg["preprocessing"]["max_width"]
        self.place = cfg["preprocessing"]["place_train" if is_train else "place_eval"]
        self.use_illum = cfg["preprocessing"]["use_illum"]
        self.use_clahe = cfg["preprocessing"]["use_clahe"]
        self.patch = cfg["preprocessing"]["patch_size"]
        self.stride = cfg["preprocessing"]["stride"]
        self.min_fg = cfg["preprocessing"]["min_foreground"]

        if len(self.writer_ids) < 2:
            raise RuntimeError("Need at least 2 writers with >=2 images each to form triplets.")
        # Total number of line images available in this dataset (train or val).
        self.num_samples = sum(len(v) for v in self.writers_dict.values())

    def __len__(self):
        # This special method tells the training loop how many batches are in one epoch.
        return max(1, self.num_samples // self.bs)

    def _line_to_K_patches(self, img_path):
        """
        This helper function converts a single line image into a stack of exactly K patches.
        - preprocess --> content-aware patches
        - if more than K: random sample K
        - if fewer than K: repeat pathes to reach K
        This ensures every line is represented by the same amount of data.
        """
        # 1. Run the full preprocessing pipeline on the line image.
        img, dbg = preprocess_line(
            img_path, target_h=self.H, max_w=self.W, place=self.place,
            use_illum=self.use_illum, use_clahe=self.use_clahe
        )
        # 2. Extract all content-aware patches from the preprocessed line.
        patches = extract_patches_content_only(
            img, dbg["content_bounds"], patch=self.patch, stride=self.stride, min_fg=self.min_fg
        )  # Shape is (N, patch, patch, 1), where N is the number of patches found.

        # 3. Force the number of patches to be exactly K.
        if patches.shape[0] >= self.K:
            # If we have more than K patches, randomly sample K of them.
            idx = np.random.choice(patches.shape[0], self.K, replace=False)
            Kpatches = patches[idx]
        else:
            # If we have fewer than K, repeat some patches until we have K.
            reps = [patches[i % patches.shape[0]] for i in range(self.K)]
            Kpatches = np.stack(reps, axis=0)

        if Kpatches.ndim == 3:
            Kpatches = Kpatches[..., None]
        return Kpatches.astype(np.float32)

    def __iter__(self):
        # This makes the class iterable, allowing a standard `for` loop.
        for _ in range(self.__len__()):
            yield self.__getitem__(0) # Yields a new random batch each time

    def __getitem__(self, idx):
        """
        This is the core method that generates one batch of random triplets.
        A,P,N: (batch, K, patch, patch, 1)
        y: dummy zeros (triplet loss ignores labels)
        """
        # Pre-allocate numpy arrays for the batch.
        A = np.zeros((self.bs, self.K, self.patch, self.patch, 1), dtype=np.float32)
        P = np.zeros_like(A); N = np.zeros_like(A)

        # For each spot in the batch, create one random triplet.
        for i in range(self.bs):
            # Anchor/positive are from the same writer.
            aw = random.choice(self.writer_ids)
            a_path, p_path = sample_two_distinct(self.writers_dict[aw], random)

            # Negative is from a different writer.
            nw = random.choice([w for w in self.writer_ids if w != aw])
            n_path = random.choice(self.writers_dict[nw])

            # Convert each line image into a stack of K patches.
            A[i] = self._line_to_K_patches(a_path)
            P[i] = self._line_to_K_patches(p_path)
            N[i] = self._line_to_K_patches(n_path)

        # The loss function doesn't use labels, so we return a dummy placeholder.
        y = np.zeros((self.bs, 1), dtype=np.float32)
        return (A, P, N), y










def main():
    # Load all hyperparameters from the YAML config file.
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    # Set random seeds for reproducibility.
    set_seeds(cfg["training"]["seed"])

    # Set the device to GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all image paths and group them by writer ID.
    train_map = index_writer_images(Path(cfg["paths"]["train_dir"]))

    # If a separate validation directory isn't provided, split the training writers
    # into a training set and a validation set (e.g., 80% train, 20% val).
    if cfg["paths"]["val_dir"] and Path(cfg["paths"]["val_dir"]).exists():
        val_map = index_writer_images(Path(cfg["paths"]["val_dir"]))
    else:
        train_map, val_map = split_writers(
            train_map,
            val_frac=cfg["training"]["val_split_by_writers"],
            seed=cfg["training"]["seed"]
        )

    print(f"Train writers: {len(train_map)} | Val writers: {len(val_map)}")

    # Create instances of our data generator for both training and validation.
    train_gen = TripletPatchGenerator(train_map, cfg, is_train=True)
    val_gen   = TripletPatchGenerator(val_map,   cfg, is_train=False)

    # --- Hierarchical Model Building ---
    # 1. Get the configuration for the models.
    patch = cfg["preprocessing"]["patch_size"]
    emb   = cfg["training"]["embedding_dim"]
    K     = cfg["training"]["patches_per_line"]

    # 2. Build the Patch Encoder: The base CNN that processes one patch.
    patch_encoder = get_patch_encoder(input_shape=(patch, patch, 1), embedding_dim=emb)
    # 3. Build the Line Encoder: Wraps the Patch Encoder to process K patches and pool them.
    line_encoder  = get_line_encoder(patch_encoder, patches_per_line=K)
    # 4. Build the Siamese Model: Uses the Line Encoder to process three lines (A, P, N).
    
    siamese, triplet_loss = build_triplet_siamese(
        line_encoder, embedding_dim=emb, margin=cfg["training"]["triplet_margin"]
    )

    # Move the final model to the selected device (GPU or CPU).
    siamese.to(device)

    # Set up the optimizer (Adam) and a learning rate scheduler.
    # The scheduler will automatically reduce the learning rate if validation loss plateaus.
    opt = Adam(siamese.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # Prepare directories and variables for saving the best model (checkpointing).
    Path(cfg["paths"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg["paths"]["siamese_weights"]
    best_val = float("inf")
    # Set up variables for early stopping.
    patience = cfg["training"]["early_stopping_patience"]
    no_improve = 0

    # --- Main Training Loop ---
    print("Starting training...")
    epochs = cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        # Set the model to training mode (enables layers like Dropout).
        siamese.train()
        running = 0.0
        # tqdm creates a smart progress bar for the loop.
        pbar = tqdm(train_gen, desc=f"Epoch {epoch}/{epochs}", leave=True)

        # --- CORRECTED TRAINING LOOP ---
        # Iterate directly over the generator to get a new random batch each time.
        for (A, P, N), _y in pbar:
            # Data comes from the generator as a NumPy array with shape (B,K,H,W,C).
            # PyTorch Conv layers expect (B,C,H,W), so we must transpose the axes.
            A = torch.from_numpy(np.transpose(A, (0,1,4,2,3))).to(device) # Shape -> (B,K,1,H,W)
            P = torch.from_numpy(np.transpose(P, (0,1,4,2,3))).to(device)
            N = torch.from_numpy(np.transpose(N, (0,1,4,2,3))).to(device)

            # --- Core Training Step ---
            # 1. Reset gradients from the previous step.
            opt.zero_grad(set_to_none=True)
            
            # Use mixed precision if a scaler is available (on GPU).
            if scaler is not None:
                # autocast runs the forward pass in float16 for speed.
                with torch.cuda.amp.autocast():
                    y_pred = siamese(A, P, N)
                    loss = triplet_loss(None, y_pred)
                # scaler manages the loss scaling to prevent underflow.
                scaler.scale(loss).backward() # Backward pass
                scaler.step(opt)              # Optimizer step
                scaler.update()               # Update scaler for next iteration
            else:
                # Standard training on CPU or without mixed precision.
                y_pred = siamese(A, P, N)   # 2. Forward pass: get model predictions.
                loss = triplet_loss(None, y_pred) # 3. Calculate the loss.
                loss.backward()             # 4. Backward pass: compute gradients.
                opt.step()                  # 5. Update model weights.

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, len(train_gen))

        # --- Validation Loop ---
        # Set the model to evaluation mode (disables layers like Dropout).
        siamese.eval()
        val_running = 0.0
        # torch.no_grad() disables gradient calculation to save memory and speed up inference.
        with torch.no_grad():
            # --- CORRECTED VALIDATION LOOP ---
            for (A, P, N), _y in val_gen:
                A = torch.from_numpy(np.transpose(A, (0,1,4,2,3))).to(device)
                P = torch.from_numpy(np.transpose(P, (0,1,4,2,3))).to(device)
                N = torch.from_numpy(np.transpose(N, (0,1,4,2,3))).to(device)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        y_pred = siamese(A, P, N)
                        vloss = triplet_loss(None, y_pred)
                else:
                    y_pred = siamese(A, P, N)
                    vloss = triplet_loss(None, y_pred)
                val_running += vloss.item()

        val_loss = val_running / max(1, len(val_gen))
        print(f"Epoch {epoch}: loss={train_loss:.6f} - val_loss={val_loss:.6f}")

        # Update the learning rate based on the validation loss.
        scheduler.step(val_loss)

        # Manual implementation of ModelCheckpoint (save best only).
        if val_loss < best_val - 1e-12:
            print(f"Epoch {epoch}: val_loss improved from {best_val if best_val!=float('inf') else 'inf'} to {val_loss:.6f}, saving model weights...")
            best_val = val_loss
            # Save the state dict (weights) of the whole siamese model and the base encoder separately.
            torch.save({'model_state_dict': siamese.state_dict()}, ckpt_path)
            



            # currently I am saving the best model here too, when I finally built the model, remove this. As here only the best check points to be saved
            
            base_path = cfg["paths"]["base_encoder_weights"]
            torch.save({'patch_encoder_state_dict': patch_encoder.state_dict()}, base_path)
            
            
            no_improve = 0

        else:
            no_improve += 1

        # Manual implementation of EarlyStopping.
        if no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            # Restore the best weights before stopping.
            state = torch.load(ckpt_path, map_location=device)
            siamese.load_state_dict(state['model_state_dict'])
            break
            
    # Final save of the patch encoder weights for inference/evaluation.
    base_path = cfg["paths"]["base_encoder_weights"]
    torch.save({'patch_encoder_state_dict': patch_encoder.state_dict()}, base_path)
    print(f"Saved patch encoder weights to {base_path}")


if __name__ == "__main__":
    main()