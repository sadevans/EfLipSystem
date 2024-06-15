import torch
import random
import torchvision


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class VideoTransform:
    def __init__(self, speed_rate=1):
        # if subset == "val" or subset == "test":
        self.pipeline = torch.nn.Sequential(
            # FunctionalModule(lambda x: x.permute(0, 3, 1, 2)),
            torchvision.transforms.Grayscale(),
            FunctionalModule(lambda x: x.squeeze(1)),
            FunctionalModule(lambda x: x.unsqueeze(-1)),
            FunctionalModule(lambda x: x if speed_rate == 1 else torch.index_select(x, dim=0, index=torch.linspace(0, x.shape[0]-1, int(x.shape[0] / speed_rate), dtype=torch.int64))),
            FunctionalModule(lambda x: x.permute(3, 0, 1, 2)),
            FunctionalModule(lambda x: x / 255.),
            torchvision.transforms.CenterCrop(88),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def __call__(self, sample):
        return self.pipeline(sample)