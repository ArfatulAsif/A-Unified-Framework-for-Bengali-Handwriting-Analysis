
"""
model_defs.py
-------------
Ensemble "Dual-Path" Fast-Track Encoder.
Combines two lightweight experts:
  1. Standard Branch: Captures micro-textures.
  2. Dilated Branch: Captures long-range stroke connectivity.
Fuses them into a single high-performance biometric embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. THE PATCH ENCODER (The Ensemble)
# ==========================================

def get_patch_encoder(input_shape=(128, 128, 1), embedding_dim=128) -> nn.Module:
    class DualPathEncoder(nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            
            # --- SHARED STEM: Initial Downsampling (128 -> 64) ---
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )

            # --- BRANCH 1: Standard Convolution (Local Textures) ---
            self.branch1 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

            # --- BRANCH 2: Dilated Convolution (Long-range Strokes) ---
            # Dilation of 2 allows the 3x3 kernel to see a 5x5 area 
            # without increasing the parameter count.
            self.branch2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

            # --- FUSION LAYER ---
            # We concatenate the 64-dim outputs into a 128-dim feature map
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, embedding_dim)

        def forward(self, x):
            x = self.stem(x)
            
            # Run branches in parallel
            f1 = self.branch1(x)
            f2 = self.branch2(x)
            
            # Combine features
            combined = torch.cat([f1, f2], dim=1)
            
            x = self.gap(combined)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            return F.normalize(x, p=2, dim=1)

    return DualPathEncoder(embedding_dim)


# ==========================================
# 2. THE LINE ENCODER
# ==========================================

def get_line_encoder(patch_encoder: nn.Module, patches_per_line: int=None) -> nn.Module:
    class FastLineEncoder(nn.Module):
        def __init__(self, patch_encoder):
            super().__init__()
            self.patch_encoder = patch_encoder

        def forward(self, x):
            # x shape: (Batch, K_patches, Channels, Height, Width)
            B, K, C, H, W = x.shape
            
            # Flatten B and K to process all patches in one forward pass
            x = x.reshape(B * K, C, H, W)
            E = self.patch_encoder(x)
            
            # Reshape back to (Batch, K, Embedding_Dim)
            E = E.reshape(B, K, -1)
            
            # Mean pooling across the K patches to represent the entire line
            pooled = E.mean(dim=1)
            return F.normalize(pooled, p=2, dim=1)
            
    return FastLineEncoder(patch_encoder)


# ==========================================
# 3. THE SIAMESE WRAPPER & LOSS
# ==========================================

def build_triplet_siamese(line_encoder: nn.Module, embedding_dim=128, margin=0.3):
    class TripletSiamese(nn.Module):
        def __init__(self, line_encoder, embedding_dim):
            super().__init__()
            self.line_encoder = line_encoder
            self.embedding_dim = embedding_dim

        def forward(self, a_in, p_in, n_in):
            # Process Anchor, Positive, and Negative branches
            return torch.cat([
                self.line_encoder(a_in), 
                self.line_encoder(p_in), 
                self.line_encoder(n_in)
            ], dim=-1)

    model = TripletSiamese(line_encoder, embedding_dim)

    def triplet_loss(_, y_pred):
        # Slice the concatenated output back into A, P, N
        a = y_pred[:, :embedding_dim]
        p = y_pred[:, embedding_dim:2*embedding_dim]
        n = y_pred[:, 2*embedding_dim:]
        
        # Calculate squared Euclidean distances
        pos_dist = torch.sum((a - p) ** 2, dim=1)
        neg_dist = torch.sum((a - n) ** 2, dim=1)
        
        # Triplet Margin Loss
        return torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()

    return model, triplet_loss



