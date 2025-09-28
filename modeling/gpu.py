


#  conda create --name "pt-gpu"

# conda activate pt-gpu

#  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

#

import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


import torch

if torch.cuda.is_available():
    print("PyTorch is running on the GPU.")
    # Check the CUDA version PyTorch was compiled with
    print(f"CUDA Version: {torch.version.cuda}")
    # Check the cuDNN version
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    # Get the name of the GPU
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is running on the CPU.")