"""
model_defs.py
-------------
Defines the core PyTorch modules for the writer verification task:
  - Patch encoder CNN (produces L2-normalized patch embeddings).      : L2 - Normalized = making lenght of each patch vector exactly same, so the vectors only differs in Direction(-->)
  - Line encoder (TimeDistributed patch encoder + mean pooling)
  - Triplet Siamese (anchor, positive, negative) with custom triplet loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_patch_encoder(input_shape=(128,128,1), embedding_dim=128) -> nn.Module:
    """
    This function builds and returns the base CNN that acts as a feature extractor.
    Its job is to take a single image patch and convert it into a dense feature vector (embedding).
    """
    class PatchEncoder(nn.Module):
        """A small but effective CNN for patch-level embeddings."""
        def __init__(self, input_shape, embedding_dim):
            super().__init__()
            H, W, C = input_shape
            assert C == 1 # This model is designed for grayscale images (1 channel).

            # --- Hierarchical Feature Extraction Layers ---
            # These layers learn progressively more complex features, from simple edges to parts of characters.

            # Block 1: Learns low-level features like edges and curves.
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2) # Halves the spatial dimensions (128x128 -> 64x64).

            # Block 2: Combines edges and curves into more complex shapes.
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2) # Reduces dimensions (64x64 -> 32x32).

            # Block 3: Combines shapes into higher-level patterns, like parts of characters.
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(2) # Reduces dimensions (32x32 -> 16x16).

            # --- Feature Interpretation and Embedding Layers ---
            # These layers take the learned features and map them to the final embedding space.

            self.flatten = nn.Flatten() # Converts the 2D feature map into a 1D vector.
            self.dropout = nn.Dropout(0.4) # Regularization layer to prevent overfitting.
            
            # nn.LazyLinear automatically infers the input size from the first forward pass.
            self.fc1 = nn.LazyLinear(256) 
            # This final linear layer projects the features down to the desired embedding dimension.
            self.fc2 = nn.Linear(256, embedding_dim)
            
            self.relu = nn.ReLU() # Standard activation function.

        def forward(self, x):
            """Defines the forward pass (the data's path through the layers)."""
            # Input x has shape: (Batch, Channels, Height, Width) -> (B, 1, H, W) : Channels = 1, only grayscale, it would be 3 for RGB
            
            # Pass through the convolutional blocks.
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.relu(self.conv3(x))
            x = self.pool3(x)
            
            # Flatten the feature map and pass through the dense layers.
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.relu(self.fc1(x))

            # L2 normalize the vector before the final projection. This can help stabilize training.
            x = F.normalize(x, p=2, dim=1)
            
            # Project to the final embedding dimension.
            x = self.fc2(x)

            # Final L2 normalization. This is crucial for similarity learning, as it ensures
            # all embeddings have a length of 1, forcing the model to focus on direction (features).
            x = F.normalize(x, p=2, dim=1)
            return x

    return PatchEncoder(input_shape, embedding_dim)


def get_line_encoder(patch_encoder: nn.Module, patches_per_line: int=None) -> nn.Module:
    """
    This function builds the Line Encoder. It wraps the Patch Encoder and adds a pooling
    step to create a single embedding for an entire line from its constituent patches.
    """
    class LineEncoder(nn.Module):
        """Wraps the patch encoder and mean-pools across K patches."""     # Mean pooling = averaging the k pathes into creating one, which represents the lines embedding
        def __init__(self, patch_encoder, patches_per_line):
            super().__init__()
            # The LineEncoder uses the previously defined PatchEncoder as its core component.
            self.patch_encoder = patch_encoder
            self.patches_per_line = patches_per_line

        def forward(self, x):
            # Input x is a stack of K patches for each item in the batch.
            # Shape: (Batch, K_patches, Channels, Height, Width) -> (B, K, 1, H, W)
            B, K, C, H, W = x.shape
            
            # To process all patches efficiently, we reshape the input so that the batch
            # and patch dimensions are combined. Shape -> (B*K, C, H, W)
            x = x.reshape(B*K, C, H, W)
            
            # Pass all B*K patches through the patch_encoder at once.
            E = self.patch_encoder(x)          # Output shape: (B*K, embedding_dim)
            
            # Reshape the output back to separate the batch and patch dimensions.
            D = E.shape[1] # This is the embedding_dim
            E = E.reshape(B, K, D)             # Shape -> (B, K, D)
            
            # Mean Pooling: Average the K patch embeddings for each line to get a single line embedding.
            pooled = E.mean(dim=1)             # Shape -> (B, D)
            
            # L2 normalize the final pooled line embedding.
            out = F.normalize(pooled, p=2, dim=1)
            return out

    return LineEncoder(patch_encoder, patches_per_line)


def build_triplet_siamese(line_encoder: nn.Module, embedding_dim=128, margin=0.2):     # by default embedding_dim = 128 and margin = 0.2, but you can change them during training
    """
    This function builds the complete Siamese network for training.
    It uses the Line Encoder three times (for A, P, N) and defines the triplet loss function.
    """
    class TripletSiamese(nn.Module):
        """Triplet network that computes and concatenates three line embeddings."""
        def __init__(self, line_encoder, embedding_dim):
            super().__init__()
            # The line_encoder is a shared component. The same instance with the same weights
            # will be used to process the anchor, positive, and negative lines.
            self.line_encoder = line_encoder
            self.embedding_dim = embedding_dim

        def forward(self, a_in, p_in, n_in):
            """Process the anchor, positive, and negative inputs."""
            a_e = self.line_encoder(a_in) # Get anchor embedding
            p_e = self.line_encoder(p_in) # Get positive embedding
            n_e = self.line_encoder(n_in) # Get negative embedding
            
            # Concatenate the three embeddings into a single tensor.
            # This is a convenient way to pass all three to the loss function.
            merged = torch.cat([a_e, p_e, n_e], dim=-1)  # Shape -> (B, 3 * embedding_dim)
            return merged

    model = TripletSiamese(line_encoder, embedding_dim)

    def triplet_loss(_, y_pred):
        """
        Custom triplet loss function. It takes the concatenated embeddings and calculates
        the loss based on the relative distances between A, P, and N.
        """
        # Split the concatenated tensor back into individual embeddings.
        a = y_pred[:, :embedding_dim]
        p = y_pred[:, embedding_dim:2*embedding_dim]
        n = y_pred[:, 2*embedding_dim:]
        
        # Calculate the squared L2 distance between Anchor and Positive.
        pos_dist = ((a - p) ** 2).sum(dim=-1)
        # Calculate the squared L2 distance between Anchor and Negative.
        neg_dist = ((a - n) ** 2).sum(dim=-1)
        
        # The core of the triplet loss formula: loss = max(0, pos_dist - neg_dist + margin)
        # It penalizes the model if the positive pair is not closer than the negative pair by at least the margin.
        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        
        # Return the average loss over the entire batch.
        return loss.mean()

    return model, triplet_loss