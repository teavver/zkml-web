import torch
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

print('start')
model = MinimalModel()
x = torch.rand(1, 48)
print(x.shape)
output = model(x)