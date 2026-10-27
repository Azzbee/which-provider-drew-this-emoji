"""Model definitions.

EmojiCNN      : the team's first baseline (3 conv blocks, no BatchNorm) kept for the ablation
EmojiCNNv2    : compact CNN from scratch with BatchNorm, 4 blocks, global average pooling (~0.4M params)
MetaFusion    : wraps any image model and concatenates a small embedding of file metadata before the classifier
build_backbone: ImageNet-pretrained backbone from timm (ConvNeXt-Tiny / ResNet18), new 7-way head
"""
from __future__ import annotations

import torch
from torch import nn


class EmojiCNN(nn.Module):
    """Original baseline (Sept 30): Conv-ReLU-MaxPool x3, GAP, dropout 0.25, linear."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, n_classes))
        self.feature_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), 1)


def conv_bn(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class EmojiCNNv2(nn.Module):
    """Compact CNN trained from scratch.

    Four stages (32-64-128-256 channels), each = Conv3x3-BN-ReLU followed by 2x2 max pooling, then global
    average pooling, dropout and one linear layer. At 96 px input the receptive field after stage 4 covers
    the whole emoji. 0.39M parameters, ~0.14 GFLOPs per image: trains in minutes on a laptop CPU.
    """

    def __init__(self, n_classes: int, widths=(32, 64, 128, 256), dropout: float = 0.3) -> None:
        super().__init__()
        layers, cin = [], 3
        for i, w in enumerate(widths):
            layers.append(conv_bn(cin, w))
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            cin = w
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.feature_dim = cin
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), 1)


class MetaFusion(nn.Module):
    """Late fusion: image embedding || MLP(file metadata) -> linear classifier.

    Used only for the ablation that measures how much the file-format signal adds on top of pixels.
    """

    def __init__(self, image_model: nn.Module, meta_dim: int, n_classes: int, meta_hidden: int = 16,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.image_model = image_model
        self.meta_net = nn.Sequential(nn.Linear(meta_dim, meta_hidden), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(image_model.feature_dim + meta_hidden, n_classes))

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([self.image_model.embed(x), self.meta_net(meta)], dim=1))


class TimmBackbone(nn.Module):
    """ImageNet-pretrained backbone with a fresh linear head; exposes embed() like the small CNNs."""

    def __init__(self, name: str, n_classes: int, pretrained: bool = True, drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0) -> None:
        super().__init__()
        import timm  # imported lazily so the CPU-only path does not need it
        kw = {"drop_path_rate": drop_path_rate} if drop_path_rate else {}
        self.body = timm.create_model(name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate, **kw)
        self.feature_dim = self.body.num_features
        self.classifier = nn.Linear(self.feature_dim, n_classes)
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.body(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def build_model(kind: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    kind = kind.lower()
    if kind == "emojicnn":
        return EmojiCNN(n_classes)
    if kind == "emojicnnv2":
        return EmojiCNNv2(n_classes)
    if kind in ("convnext_tiny", "resnet18", "resnet34", "efficientnet_b0", "convnext_small"):
        return TimmBackbone(kind, n_classes, pretrained=pretrained)
    raise ValueError(f"unknown model kind: {kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
