"""
model_defs.py
-------------
High-Speed "Fast-Track" Encoder.
Optimized for minimum training time per epoch:
  - Immediate downsampling to reduce spatial FLOPs.
  - Efficient 3-block architecture.
  - Optimized for NVIDIA Tensor Cores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_patch_encoder(input_shape=(128,128,1), embedding_dim=128) -> nn.Module:
    class FastPatchEncoder(nn.Module):
        def __init__(self, input_shape, embedding_dim):
            super().__init__()
            
            # --- Block 1: Immediate Downsampling (128 -> 64) ---
            # Using a stride of 2 here instead of MaxPool later saves one full operation
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            
            # --- Block 2: Feature Extraction (64 -> 32) ---
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            
            # --- Block 3: Style Bottleneck (32 -> 16) ---
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(128)
            
            self.act = nn.ReLU(inplace=True) # ReLU is slightly faster than LeakyReLU
            
            # Spatial Reduction to 1x1 vector
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, embedding_dim)

        def forward(self, x):
            x = self.act(self.bn1(self.conv1(x)))
            x = self.act(self.bn2(self.conv2(x)))
            x = self.act(self.bn3(self.conv3(x)))
            
            x = self.gap(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            return F.normalize(x, p=2, dim=1)

    return FastPatchEncoder(input_shape, embedding_dim)

def get_line_encoder(patch_encoder: nn.Module, patches_per_line: int=None) -> nn.Module:
    class FastLineEncoder(nn.Module):
        def __init__(self, patch_encoder, patches_per_line):
            super().__init__()
            self.patch_encoder = patch_encoder

        def forward(self, x):
            B, K, C, H, W = x.shape
            x = x.reshape(B*K, C, H, W)
            E = self.patch_encoder(x)
            
            # Mean pooling across K patches
            E = E.reshape(B, K, -1)
            pooled = E.mean(dim=1)
            return F.normalize(pooled, p=2, dim=1)
            
    return FastLineEncoder(patch_encoder, patches_per_line)

def build_triplet_siamese(line_encoder: nn.Module, embedding_dim=128, margin=0.3):
    class TripletSiamese(nn.Module):
        def __init__(self, line_encoder, embedding_dim):
            super().__init__()
            self.line_encoder = line_encoder
            self.embedding_dim = embedding_dim

        def forward(self, a_in, p_in, n_in):
            # Parallel branch execution
            return torch.cat([self.line_encoder(a_in), 
                              self.line_encoder(p_in), 
                              self.line_encoder(n_in)], dim=-1)

    model = TripletSiamese(line_encoder, embedding_dim)

    def triplet_loss(_, y_pred):
        a = y_pred[:, :embedding_dim]
        p = y_pred[:, embedding_dim:2*embedding_dim]
        n = y_pred[:, 2*embedding_dim:]
        
        # Fast vector distance calculation
        pos_dist = torch.sum((a - p) ** 2, dim=1)
        neg_dist = torch.sum((a - n) ** 2, dim=1)
        
        return torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()

    return model, triplet_loss