"""
Modell-Definitionen: SimpleCNN und angepasstes ResNet-18 für CIFAR-10.
"""

import torch
import torch.nn as nn
import torchvision.models as models

import config


class SimpleCNN(nn.Module):
    """
    Einfaches CNN mit 3 Conv-Blöcken und 2 FC-Layern.
    Geschätzte Parameter: ~350k
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(config.DROPOUT_CONV),

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(config.DROPOUT_CONV),

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(config.DROPOUT_CONV),
        )

        # Nach 3x MaxPool(2,2): 32 → 16 → 8 → 4, also 128 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_FC),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 angepasst für CIFAR-10 (32x32):
    - Erste Conv: 3x3 statt 7x7, stride=1, padding=1
    - Kein MaxPool nach erster Conv (Bild zu klein)
    - FC: 512 → 10
    Geschätzte Parameter: ~11M
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        self.resnet = models.resnet18(weights=None)

        # Erste Conv-Schicht anpassen: 3x3 statt 7x7
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # MaxPool entfernen (Identity)
        self.resnet.maxpool = nn.Identity()
        # FC-Layer anpassen
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


def create_model(name: str = config.MODEL_NAME) -> nn.Module:
    """Erstellt das gewünschte Modell."""
    if name == "SimpleCNN":
        return SimpleCNN()
    elif name == "ResNet18":
        return ResNet18CIFAR()
    else:
        raise ValueError(f"Unbekanntes Modell: {name}")


def count_parameters(model: nn.Module) -> int:
    """Zählt die trainierbaren Parameter."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
