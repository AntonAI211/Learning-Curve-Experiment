"""
Trainingsloop mit Early Stopping und LR-Scheduling.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config


def create_optimizer(model: nn.Module, name: str = config.OPTIMIZER_NAME):
    """Erstellt den Optimizer basierend auf Konfiguration."""
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=config.ADAM_LR)
    elif name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.SGD_LR,
            momentum=config.SGD_MOMENTUM,
            weight_decay=config.SGD_WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unbekannter Optimizer: {name}")


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple:
    """Berechnet Loss und Accuracy auf einem DataLoader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            # Speicher freigeben während Evaluation
            del inputs, targets, outputs, loss, predicted

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device = config.DEVICE,
    optimizer_name: str = config.OPTIMIZER_NAME,
) -> dict:
    """
    Trainiert das Modell mit Early Stopping.
    Gibt ein Dictionary mit allen relevanten Metriken zurück.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, optimizer_name)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(config.MAX_EPOCHS):
        # ── Training ────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            # Speicher aggressiv freigeben
            del inputs, targets, outputs, loss, predicted
            if device.type == "mps":
                torch.mps.empty_cache()

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # ── Validation ──────────────────────────────────────────────
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # ── LR Scheduler ────────────────────────────────────────────
        scheduler.step(val_loss)

        # ── Progress Output ─────────────────────────────────────────
        print(f"  Epoche {epoch+1}/{config.MAX_EPOCHS}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.2%}, Patience={patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

        # ── Early Stopping ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                break

    training_time = time.time() - start_time
    epochs_trained = epoch + 1

    # ── Bestes Modell laden und auf Test evaluieren ──────────────────
    model.load_state_dict(best_model_state)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    return {
        "final_test_accuracy": test_accuracy,
        "final_test_loss": test_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
        "training_time_seconds": round(training_time, 2),
        "train_loss_per_epoch": train_losses,
        "train_accuracy_per_epoch": train_accuracies,
        "val_loss_per_epoch": val_losses,
        "val_accuracy_per_epoch": val_accuracies,
    }
