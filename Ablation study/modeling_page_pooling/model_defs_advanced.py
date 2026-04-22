"""
model_defs_vit.py
-----------------
Modern SOTA Aggregation (2023-2024 Paradigm):
Dual-Path CNN + Transformer [CLS] Token Aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. THE PATCH ENCODER (DPE-Net Base)
# ==========================================
# We keep the base CNN identical to ensure the ablation study 
# strictly isolates the effect of the pooling strategy.
def get_patch_encoder(input_shape=(128, 128, 1), embedding_dim=512) -> nn.Module:
    class DualPathEncoder(nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.branch1 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, embedding_dim)

        def forward(self, x):
            x = self.stem(x)
            f1 = self.branch1(x)
            f2 = self.branch2(x)
            combined = torch.cat([f1, f2], dim=1)
            x = self.gap(combined)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return F.normalize(x, p=2, dim=1)

    return DualPathEncoder(embedding_dim)

# ==========================================
# 2. TRANSFORMER [CLS] TOKEN POOLING
# ==========================================
def get_page_encoder(patch_encoder: nn.Module, embedding_dim=512) -> nn.Module:
    class ViTPageEncoder(nn.Module):
        def __init__(self, patch_encoder, emb_dim, num_heads=8, num_layers=2):
            super().__init__()
            self.patch_encoder = patch_encoder
            self.emb_dim = emb_dim
            
            # 1. The Learnable [CLS] Token
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
            
            # 2. The Multi-Head Self-Attention Engine
            # Note: We omit positional embeddings explicitly because the K=32 patches 
            # are randomly sampled. This forces the Transformer to act as a 
            # permutation-invariant set aggregator (a true "bag of patches").
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, 
                nhead=num_heads, 
                dim_feedforward=emb_dim * 4, 
                dropout=0.1, 
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(self, x):
            # x shape: (Batch, K_Patches, Channels, Height, Width)
            B, K, C, H, W = x.shape
            
            # Extract Patch Embeddings
            x = x.reshape(B * K, C, H, W)
            E = self.patch_encoder(x)
            E = E.reshape(B, K, self.emb_dim)  # Shape: (B, K, Emb_Dim)
            
            # Append the [CLS] Token to the sequence
            # Expand cls_token to match batch size: (B, 1, Emb_Dim)
            cls_tokens = self.cls_token.expand(B, -1, -1) 
            
            # Concatenate -> Sequence length becomes K + 1
            E_seq = torch.cat((cls_tokens, E), dim=1) # Shape: (B, K+1, Emb_Dim)
            
            # Pass through Transformer
            out_seq = self.transformer(E_seq)
            
            # SINK: Extract ONLY the [CLS] token (Index 0) as the global identity vector
            cls_out = out_seq[:, 0, :]
            
            # Final L2 Normalization for Cosine distance
            return F.normalize(cls_out, p=2, dim=1)
            
    return ViTPageEncoder(patch_encoder, embedding_dim)

# ==========================================
# 3. THE SIAMESE WRAPPER & LOSS
# ==========================================
def build_triplet_siamese(page_encoder: nn.Module, embedding_dim=512, margin=0.4):
    class TripletSiamese(nn.Module):
        def __init__(self, page_encoder, embedding_dim):
            super().__init__()
            self.page_encoder = page_encoder
            self.embedding_dim = embedding_dim

        def forward(self, a_in, p_in, n_in):
            return torch.cat([
                self.page_encoder(a_in), 
                self.page_encoder(p_in), 
                self.page_encoder(n_in)
            ], dim=-1)

    model = TripletSiamese(page_encoder, embedding_dim)

    def triplet_loss(_, y_pred):
        a = y_pred[:, :embedding_dim]
        p = y_pred[:, embedding_dim:2*embedding_dim]
        n = y_pred[:, 2*embedding_dim:]
        pos_dist = torch.sum((a - p) ** 2, dim=1)
        neg_dist = torch.sum((a - n) ** 2, dim=1)
        return torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()

    return model, triplet_loss