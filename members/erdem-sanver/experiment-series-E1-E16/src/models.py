"""Model factory: a small from-scratch CNN, or any timm backbone (optionally pretrained / frozen)."""
import timm
import torch
from torch import nn


class SimpleCNN(nn.Module):
    """3 conv blocks (conv-bn-relu x2 + maxpool) -> global avg pool -> linear. ~0.3M params."""

    def __init__(self, in_ch: int, n_classes: int, width: int = 32) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(block(in_ch, width), block(width, width * 2), block(width * 2, width * 4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(width * 4, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(name: str, in_ch: int, n_classes: int, pretrained: bool, freeze: bool) -> nn.Module:
    if name == "simple_cnn":
        if pretrained or freeze:
            raise ValueError("simple_cnn has no pretrained weights")
        return SimpleCNN(in_ch, n_classes)
    model = timm.create_model(name, pretrained=pretrained, num_classes=n_classes, in_chans=in_ch)
    if freeze:
        head = model.get_classifier()
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
