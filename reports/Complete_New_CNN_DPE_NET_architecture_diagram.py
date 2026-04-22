# python -m reports.Complete_New_CNN_Ensemble_architecture_diagram


import torch

# Import your functions
from modeling.model_defs import get_patch_encoder

# Create model
model = get_patch_encoder(input_shape=(128, 128, 1), embedding_dim=128)
model.eval()  # IMPORTANT for export

# Create dummy input (Batch, Channels, Height, Width)
dummy_input = torch.randn(1, 1, 128, 128)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "architecture_diagram.onnx",
    input_names=["input"],
    output_names=["embedding"],
    opset_version=11,   # safe default
    dynamic_axes={
        "input": {0: "batch_size"},
        "embedding": {0: "batch_size"}
    }
)

print("Model exported to architecture_diagram.onnx")