import torch, sys, platform, os
# import numpy as np
import torch.nn as nn

class MinimalModel(nn.Module):
    def __init__(self):
        super(MinimalModel, self).__init__()
        self.d1 = nn.Linear(48, 48)

    def forward(self, x):
        print("Input:", x.shape, x.dtype, x.min().item(), x.max().item())
        x = self.d1(x)
        print("Output:", x.shape, x.dtype, x.min().item(), x.max().item())
        return x

# print(f"Python version: {sys.version}")
# print(f"Platform: {platform.platform()}")
# print(f"PyTorch version: {torch.__version__}")
# print(f"NumPy version: {np.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
# print(f"CPU count: {os.cpu_count()}")
# print(f"PyTorch default dtype: {torch.get_default_dtype()}")

model = MinimalModel()
x = torch.rand(1, 48)
output = model(x)