# train.py
# --------
# Training script for high-speed writer verification.
# Tracks: Params, FLOPs, Average Epoch Time, and Best Epoch.

import yaml, random, os, time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

# Project modules
from data.dataset import index_writer_images, split_writers, sample_two_distinct
from preprocessor.pipeline import preprocess_line, extract_patches_content_only
from modeling.model_defs import get_patch_encoder, get_line_encoder, build_triplet_siamese


def set_seeds(seed: int):
    """Set all RNG seeds for reproducibility."""
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if torch.cuda.is_available():
    os.environ["TORCH_ALLOW_TF32"] = "1"
    scaler = torch.cuda.amp.GradScaler()
    print("Mixed precision ENABLED (float16).")
else:
    scaler = None
    print("Mixed precision DISABLED (no GPU).")


class TripletPatchGenerator:
    def __init__(self, writers_dict, cfg, is_train=True):
        self.writers_dict = {w: v for w, v in writers_dict.items() if len(v) >= 2}
        self.writer_ids = list(self.writers_dict.keys())
        self.cfg = cfg
        self.is_train = is_train

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
        self.num_samples = sum(len(v) for v in self.writers_dict.values())

    def __len__(self):
        return max(1, self.num_samples // self.bs)

    def _line_to_K_patches(self, img_path):
        img, dbg = preprocess_line(
            img_path, target_h=self.H, max_w=self.W, place=self.place,
            use_illum=self.use_illum, use_clahe=self.use_clahe
        )
        patches = extract_patches_content_only(
            img, dbg["content_bounds"], patch=self.patch, stride=self.stride, min_fg=self.min_fg
        )

        if patches.shape[0] >= self.K:
            idx = np.random.choice(patches.shape[0], self.K, replace=False)
            Kpatches = patches[idx]
        else:
            reps = [patches[i % patches.shape[0]] for i in range(self.K)]
            Kpatches = np.stack(reps, axis=0)

        if Kpatches.ndim == 3:
            Kpatches = Kpatches[..., None]
        return Kpatches.astype(np.float32)

    def __iter__(self):
        for _ in range(self.__len__()):
            yield self.__getitem__(0)

    def __getitem__(self, idx):
        A = np.zeros((self.bs, self.K, self.patch, self.patch, 1), dtype=np.float32)
        P = np.zeros_like(A); N = np.zeros_like(A)

        for i in range(self.bs):
            aw = random.choice(self.writer_ids)
            a_path, p_path = sample_two_distinct(self.writers_dict[aw], random)
            nw = random.choice([w for w in self.writer_ids if w != aw])
            n_path = random.choice(self.writers_dict[nw])

            A[i] = self._line_to_K_patches(a_path)
            P[i] = self._line_to_K_patches(p_path)
            N[i] = self._line_to_K_patches(n_path)

        y = np.zeros((self.bs, 1), dtype=np.float32)
        return (A, P, N), y


def main():
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seeds(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_map = index_writer_images(Path(cfg["paths"]["train_dir"]))

    if cfg["paths"]["val_dir"] and Path(cfg["paths"]["val_dir"]).exists():
        val_map = index_writer_images(Path(cfg["paths"]["val_dir"]))
    else:
        train_map, val_map = split_writers(
            train_map,
            val_frac=cfg["training"]["val_split_by_writers"],
            seed=cfg["training"]["seed"]
        )

    print(f"Train writers: {len(train_map)} | Val writers: {len(val_map)}")

    train_gen = TripletPatchGenerator(train_map, cfg, is_train=True)
    val_gen   = TripletPatchGenerator(val_map,   cfg, is_train=False)

    patch = cfg["preprocessing"]["patch_size"]
    emb   = cfg["training"]["embedding_dim"]
    K     = cfg["training"]["patches_per_line"]

    # Model Building
    patch_encoder = get_patch_encoder(input_shape=(patch, patch, 1), embedding_dim=emb)

    # --- FIXED BENCHMARKING: Overcoming Lazy Initialization ---
    if FlopCountAnalysis is not None:
        print("\n" + "="*50)
        print("SOTA BENCHMARK: Architectural Complexity (Per Patch)")
        
        patch_encoder.eval() # Ensure dropout/batchnorm act predictably
        dummy_input = torch.randn(1, 1, patch, patch)
        
        # ---> THIS FIXES THE CRASH: Push a dummy image to build the weights <---
        with torch.no_grad():
            _ = patch_encoder(dummy_input)
        
        flops = FlopCountAnalysis(patch_encoder, dummy_input)
        # Suppress warnings to keep the console clean
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        
        total_params = sum(p.numel() for p in patch_encoder.parameters())
        
        total_flops = flops.total()
        if total_flops >= 1e9:
            flops_str = f"{total_flops / 1e9:.2f} G"
        elif total_flops >= 1e6:
            flops_str = f"{total_flops / 1e6:.2f} M"
        else:
            flops_str = f"{total_flops}"

        print(f"Total Parameters: {total_params / 1e6:.4f} M")
        print(f"FLOPs (MACs):      {flops_str}")
        print("="*50 + "\n")
        patch_encoder.train() # Return to training mode

    line_encoder  = get_line_encoder(patch_encoder, patches_per_line=K)
    siamese, triplet_loss = build_triplet_siamese(
        line_encoder, embedding_dim=emb, margin=cfg["training"]["triplet_margin"]
    )

    siamese.to(device)

    opt = Adam(siamese.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    ckpt_path = cfg["paths"]["chk_points_weights"]
    
    # Early Stopping & Best Epoch tracking setup
    best_val = float("inf")
    best_epoch = 0
    patience = cfg["training"]["early_stopping_patience"]
    no_improve = 0
    epoch_times = []

    print("Starting training...")
    epochs = cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        siamese.train()
        running = 0.0
        pbar = tqdm(train_gen, desc=f"Epoch {epoch}/{epochs}", leave=True)

        for (A, P, N), _y in pbar:
            A = torch.from_numpy(np.transpose(A, (0,1,4,2,3))).to(device) 
            P = torch.from_numpy(np.transpose(P, (0,1,4,2,3))).to(device)
            N = torch.from_numpy(np.transpose(N, (0,1,4,2,3))).to(device)

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    y_pred = siamese(A, P, N)
                    loss = triplet_loss(None, y_pred)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                y_pred = siamese(A, P, N)
                loss = triplet_loss(None, y_pred)
                loss.backward()
                opt.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, len(train_gen))

        # Validation
        siamese.eval()
        val_running = 0.0
        with torch.no_grad():
            for (A, P, N), _y in val_gen:
                A = torch.from_numpy(np.transpose(A, (0,1,4,2,3))).to(device)
                P = torch.from_numpy(np.transpose(P, (0,1,4,2,3))).to(device)
                N = torch.from_numpy(np.transpose(N, (0,1,4,2,3))).to(device)
                y_pred = siamese(A, P, N)
                vloss = triplet_loss(None, y_pred)
                val_running += vloss.item()

        val_loss = val_running / max(1, len(val_gen))
        
        # Timing
        epoch_end = time.time()
        duration = epoch_end - epoch_start
        epoch_times.append(duration)
        
        print(f"Epoch {epoch}: loss={train_loss:.6f} | val_loss={val_loss:.6f} | time={duration/60:.2f}m")

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_epoch = epoch  # Update the best epoch record
            torch.save({'model_state_dict': siamese.state_dict()}, ckpt_path)
            base_path = cfg["paths"]["base_encoder_weights"]
            torch.save({'patch_encoder_state_dict': patch_encoder.state_dict()}, base_path)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping triggered.")
            break
            
    # Final Benchmark Report
    avg_epoch = np.mean(epoch_times) / 60
    print("\n" + "="*50)
    print(f"AVG EPOCH TIME: {avg_epoch:.2f} minutes")
    print(f"BEST EPOCH:     {best_epoch} (val_loss: {best_val:.6f})")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()