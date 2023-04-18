import torch
import torch.nn as nn


class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std
