# train_advanced_flat.py
# ----------------------
# Trains the SOTA Attention+GeM Flat architecture.
# Includes rigorous VRAM tracking and Per-Patch FLOP/Parameter Benchmarking.

import yaml, random, os, time, cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- FVCORE IMPORT FOR BENCHMARKING ---
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

# Ensure build_triplet_siamese is properly added to your model_defs_advanced.py!
from modeling_page_pooling.model_defs_advanced import get_patch_encoder, get_page_encoder, build_triplet_siamese


def print_writer_stats(split_name: str, writer_map: dict):
    """Prints a formatted table of Writer IDs and their respective page counts."""
    print(f"\n{'-'*45}")
    print(f" DATASET: {split_name} ({len(writer_map)} Writers)")
    print(f"{'-'*45}")
    print(f"{'Writer ID':<30} | {'Page Count':<10}")
    print(f"{'-'*45}")
    
    total_pages = 0
    for w_id in sorted(writer_map.keys()):
        count = len(writer_map[w_id])
        total_pages += count
        print(f"{w_id:<30} | {count}")
        
    print(f"{'-'*45}")
    print(f"{'TOTAL PAGES':<30} | {total_pages}")
    print(f"{'-'*45}\n")


def index_page_writers(root: Path) -> Dict[str, List[Path]]:
    root = Path(root)
    out = {}
    if not root.exists(): return out
    for writer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            imgs.extend(sorted(writer_dir.glob(ext)))
        if imgs: out[writer_dir.name] = imgs
    return out


def split_writers(all_writers: Dict[str, List[Path]], val_frac: float = 0.2, seed: int = 42):
    writers = list(all_writers.keys())
    rnd = random.Random(seed)
    rnd.shuffle(writers)
    if len(writers) <= 1: return all_writers, {}
    n_val = max(1, int(round(len(writers) * val_frac)))
    val_ids = set(writers[:n_val])
    train = {w: all_writers[w] for w in writers if w not in val_ids}
    val = {w: all_writers[w] for w in writers if w in val_ids}
    return train, val


def sample_two_distinct(items: List[Path], rnd: random.Random):
    return rnd.sample(items, 2)


def set_seeds(seed: int):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


if torch.cuda.is_available():
    os.environ["TORCH_ALLOW_TF32"] = "1"
    scaler = torch.cuda.amp.GradScaler()
    print("[INFO] Mixed precision ENABLED (float16).")
else:
    scaler = None
    print("[INFO] Mixed precision DISABLED.")


class TripletPagePatchGenerator:
    def __init__(self, writers_dict, cfg, is_train=True):
        self.writers_dict = {w: v for w, v in writers_dict.items() if len(v) >= 2}
        self.writer_ids = list(self.writers_dict.keys())
        self.cfg = cfg
        self.bs = cfg["training"]["batch_size"]
        # Explicitly setting K=32 to evaluate max capacity of the 8GB RTX 3050 limit
        self.K = 32
        self.patch_size = cfg["preprocessing"].get("patch_size", 128)
        self.min_fg = cfg["preprocessing"].get("min_foreground", 0.04)
        self.max_dim = cfg.get("segment_lines", {}).get("max_dim", 1024)
        self.num_samples = sum(len(v) for v in self.writers_dict.values())

    def __len__(self):
        return max(1, self.num_samples // self.bs)

    def _page_to_K_patches(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        if max(h, w) > self.max_dim:
            scale = self.max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if self.cfg["preprocessing"].get("use_clahe", True):
            img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
            
        thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        fg_mask = 1.0 - (thr.astype("float32") / 255.0)
        img_f = img.astype("float32") / 255.0

        H, W = img_f.shape
        valid_patches = []
        stride = self.cfg["preprocessing"].get("stride", 56)
        
        for y in range(0, max(1, H - self.patch_size + 1), stride):
            for x in range(0, max(1, W - self.patch_size + 1), stride):
                y_end, x_end = min(y + self.patch_size, H), min(x + self.patch_size, W)
                p_fg = fg_mask[y:y_end, x:x_end]
                if (np.mean(p_fg) if p_fg.size > 0 else 0) >= self.min_fg:
                    patch = img_f[y:y_end, x:x_end]
                    if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                        patch = cv2.copyMakeBorder(patch, 0, self.patch_size - patch.shape[0], 
                                                   0, self.patch_size - patch.shape[1], 
                                                   cv2.BORDER_CONSTANT, value=1.0)
                    valid_patches.append(patch)

        if not valid_patches:
            valid_patches = [np.ones((self.patch_size, self.patch_size), dtype=np.float32)]
            
        if len(valid_patches) >= self.K:
            idx = np.random.choice(len(valid_patches), self.K, replace=False)
            Kpatches = np.stack([valid_patches[i] for i in idx], axis=0)
        else:
            Kpatches = np.stack([valid_patches[i % len(valid_patches)] for i in range(self.K)], axis=0)

        return np.expand_dims(Kpatches, axis=-1).astype(np.float32)

    def __iter__(self):
        for _ in range(self.__len__()): yield self.__getitem__(0)

    def __getitem__(self, idx):
        A = np.zeros((self.bs, self.K, self.patch_size, self.patch_size, 1), dtype=np.float32)
        P, N = np.zeros_like(A), np.zeros_like(A)
        for i in range(self.bs):
            aw = random.choice(self.writer_ids)
            a_path, p_path = sample_two_distinct(self.writers_dict[aw], random)
            nw = random.choice([w for w in self.writer_ids if w != aw])
            n_path = random.choice(self.writers_dict[nw])
            A[i] = self._page_to_K_patches(a_path)
            P[i] = self._page_to_K_patches(p_path)
            N[i] = self._page_to_K_patches(n_path)
        return (A, P, N), np.zeros((self.bs, 1), dtype=np.float32)


def main():
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seeds(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = Path(cfg["paths"]["train_dir"])
    train_map = index_page_writers(train_dir)
    train_map, val_map = split_writers(train_map, val_frac=cfg["training"]["val_split_by_writers"], seed=cfg["training"]["seed"])

    # Print the writer stats
    print_writer_stats("TRAINING SET", train_map)
    print_writer_stats("VALIDATION SET", val_map)

    train_gen = TripletPagePatchGenerator(train_map, cfg, is_train=True)
    val_gen   = TripletPagePatchGenerator(val_map, cfg, is_train=False)

    patch = cfg["preprocessing"]["patch_size"]
    emb   = cfg["training"]["embedding_dim"]
    
    # Init Advanced Model
    patch_encoder = get_patch_encoder(input_shape=(patch, patch, 1), embedding_dim=emb)
    page_encoder  = get_page_encoder(patch_encoder, embedding_dim=emb)
    siamese, triplet_loss = build_triplet_siamese(page_encoder, embedding_dim=emb, margin=cfg["training"]["triplet_margin"])
    siamese.to(device)

    # ============================================================
    # ARCHITECTURAL COMPLEXITY BENCHMARK (PARAMS & FLOPS PER PATCH)
    # ============================================================
    if FlopCountAnalysis is not None:
        print("\n" + "="*50)
        print(" SOTA BENCHMARK: Architectural Complexity (Per Patch)")
        
        patch_encoder.eval() # Ensure dropout/batchnorm act predictably
        
        # Create a dummy input representing EXACTLY ONE patch (Batch=1, C=1, H=128, W=128)
        dummy_input_patch = torch.randn(1, 1, patch, patch).to(device)
        
        # Force lazy initialization to build weights
        with torch.no_grad():
            _ = patch_encoder(dummy_input_patch)
            
        flops = FlopCountAnalysis(patch_encoder, dummy_input_patch)
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
        print(f"FLOPs (MACs):      {flops_str} (per 1 patch)")
        print("="*50 + "\n")
        
        patch_encoder.train() # Return to training mode
    # ============================================================

    opt = Adam(siamese.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    ckpt_path = cfg["paths"]["chk_points_weights"]
    base_path = cfg["paths"]["base_encoder_weights"]
    
    best_val = float("inf")
    best_epoch = 0
    patience = cfg["training"]["early_stopping_patience"]
    no_improve = 0
    epoch_times = []

    print("\n" + "="*50)
    print(" STARTING SOTA FLAT PAGE TRAINING (Attn+GeM)")
    print("="*50)
    
    epochs = cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        siamese.train()
        running_train_loss = 0.0
        
        # --- MEASURE AND PRINT BASELINE VRAM BEFORE TRAINING BATCHES ---
        print(f"\n--- STARTING EPOCH {epoch}/{epochs} ---")
        if torch.cuda.is_available():
            baseline_vram = torch.cuda.memory_allocated(device) / (1024 ** 2)
            print(f"[*] Baseline VRAM Allocated: {baseline_vram:.0f} MB")
            torch.cuda.reset_peak_memory_stats(device)
            
        train_pbar = tqdm(train_gen, desc=f"Epoch {epoch}/{epochs} [TRAIN]", leave=False)
        for (A, P, N), _ in train_pbar:
            A = torch.from_numpy(np.transpose(A, (0,1,4,2,3))).to(device) 
            P = torch.from_numpy(np.transpose(P, (0,1,4,2,3))).to(device)
            N = torch.from_numpy(np.transpose(N, (0,1,4,2,3))).to(device)

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = triplet_loss(None, siamese(A, P, N))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = triplet_loss(None, siamese(A, P, N))
                loss.backward()
                opt.step()

            running_train_loss += loss.item()
            
            # --- MEASURE PEAK VRAM FOR THE PROGRESS BAR ---
            peak_vram = 0
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            
            train_pbar.set_postfix(
                train_loss=f"{loss.item():.4f}", 
                peak_vram=f"{peak_vram:.0f}MB"
            )

        train_loss_final = running_train_loss / max(1, len(train_gen))

        siamese.eval()
        running_val_loss = 0.0
        val_pbar = tqdm(val_gen, desc=f"Epoch {epoch}/{epochs} [VAL]", leave=False)
        with torch.no_grad():
            for (A, P, N), _ in val_pbar:
                A = torch.from_numpy(np.transpose(A, (0,1,4,2,3))).to(device)
                P = torch.from_numpy(np.transpose(P, (0,1,4,2,3))).to(device)
                N = torch.from_numpy(np.transpose(N, (0,1,4,2,3))).to(device)
                vloss = triplet_loss(None, siamese(A, P, N)).item()
                running_val_loss += vloss
                val_pbar.set_postfix(val_loss=f"{vloss:.4f}")

        val_loss_final = running_val_loss / max(1, len(val_gen))
        duration = time.time() - epoch_start
        epoch_times.append(duration)

        print(f"\n--- EPOCH {epoch} SUMMARY ---")
        print(f"Train Loss : {train_loss_final:.6f} | Val Loss : {val_loss_final:.6f} | Time: {duration/60:.2f}m")
        
        scheduler.step(val_loss_final)

        if val_loss_final < best_val - 1e-12:
            best_val = val_loss_final
            best_epoch = epoch
            torch.save({'model_state_dict': siamese.state_dict()}, ckpt_path)
            # Save the full page encoder to capture Attention and GeM weights
            torch.save({'page_encoder_state_dict': page_encoder.state_dict()}, base_path)
            no_improve = 0
            print("Status     : [SAVED BEST MODEL]")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("\n[!] EARLY STOPPING TRIGGERED [!]")
                break

if __name__ == "__main__":
    main()