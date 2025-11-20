import torch.nn as nn
import torch.nn.functional as F

class hrelu(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out