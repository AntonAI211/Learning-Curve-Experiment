"""
Hauptschleife: Iteriert über alle (n, k)-Kombinationen,
trainiert jeweils ein Modell und speichert die Ergebnisse.
"""

import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

import config
from dataset import (
    create_dataloaders,
    get_subset_indices,
    load_cifar10,
    split_train_val,
)
from models import count_parameters, create_model
from train import train_model


def set_seed(seed: int):
    """Setzt alle Seeds für Reproduzierbarkeit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS hat keine deterministic-Option, aber manual_seed reicht


def load_existing_results(filepath: str) -> list:
    """Lädt bereits vorhandene Ergebnisse (Checkpoint-System)."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []


def is_already_done(results: list, n: int, k: int, model_name: str) -> bool:
    """Prüft, ob ein (n, k, model)-Run bereits abgeschlossen ist."""
    for r in results:
        if (
            r["training_size"] == n
            and r["partition_index"] == k
            and r["model"] == model_name
        ):
            return True
    return False


def save_results(results: list, filepath: str):
    """Speichert Ergebnisse als JSON."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def aggregate_results(results: list, output_path: str):
    """Aggregiert Ergebnisse pro Trainingsgröße und speichert als CSV."""
    df = pd.DataFrame(results)
    agg = (
        df.groupby("training_size")["final_test_accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    agg.columns = ["n", "mean_accuracy", "std_accuracy", "min_accuracy",
                    "max_accuracy", "num_runs"]
    # Für n=48000 mit nur 1 Run: std = 0
    agg["std_accuracy"] = agg["std_accuracy"].fillna(0.0)
    agg.to_csv(output_path, index=False)
    print(f"\nAggregierte Ergebnisse gespeichert: {output_path}")
    print(agg.to_string(index=False))
    return agg


def run_experiment():
    """Führt das gesamte Experiment durch."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    results_path = os.path.join(config.RESULTS_DIR, config.INDIVIDUAL_RESULTS_FILE)
    agg_path = os.path.join(config.RESULTS_DIR, config.AGGREGATED_RESULTS_FILE)

    # ── Bestehende Ergebnisse laden (Checkpoint) ────────────────────
    all_results = load_existing_results(results_path)
    if all_results:
        print(f"Fortgesetzt: {len(all_results)} vorhandene Runs gefunden.")

    # ── Dataset vorbereiten ─────────────────────────────────────────
    print("Lade CIFAR-10...")
    train_dataset, test_dataset = load_cifar10()
    train_indices, val_indices = split_train_val(train_dataset)

    # Val Dataset EINMAL laden (ohne Augmentation) - wird für alle Runs wiederverwendet
    import torchvision
    from dataset import get_transforms
    val_dataset = torchvision.datasets.CIFAR10(
        root=config.DATASET_ROOT, train=True, download=False,
        transform=get_transforms(train=False),
    )

    print(f"Train-Pool: {len(train_indices)}, Val: {len(val_indices)}, "
          f"Test: {len(test_dataset)}")

    # ── Modellinfo ──────────────────────────────────────────────────
    sample_model = create_model(config.MODEL_NAME)
    n_params = count_parameters(sample_model)
    print(f"Modell: {config.MODEL_NAME} ({n_params:,} Parameter)")
    print(f"Optimizer: {config.OPTIMIZER_NAME}")
    print(f"Device: {config.DEVICE}")
    del sample_model

    # ── Hauptschleife ───────────────────────────────────────────────
    total_runs = sum(
        1 if n == config.TRAIN_SIZE else config.NUM_PARTITIONS
        for n in config.TRAINING_SIZES
    )
    completed = len(all_results)
    print(f"\nStarte Experiment: {total_runs} Runs total, "
          f"{completed} bereits abgeschlossen.\n")

    experiment_start = time.time()

    for n in config.TRAINING_SIZES:
        num_k = 1 if n == config.TRAIN_SIZE else config.NUM_PARTITIONS

        for k in range(num_k):
            # ── Checkpoint-Check ────────────────────────────────────
            if is_already_done(all_results, n, k, config.MODEL_NAME):
                continue

            seed = config.get_seed(n, k)
            set_seed(seed)

            # ── Subset erstellen ────────────────────────────────────
            subset_indices = get_subset_indices(train_indices, n, k)

            train_loader, val_loader, test_loader = create_dataloaders(
                train_dataset, test_dataset, val_dataset,
                train_indices, val_indices, subset_indices,
            )

            # ── Modell erstellen und trainieren ─────────────────────
            model = create_model(config.MODEL_NAME)

            print(f"[n={n:>5}, k={k}] Training startet (seed={seed})...")

            metrics = train_model(
                model, train_loader, val_loader, test_loader,
                device=config.DEVICE,
                optimizer_name=config.OPTIMIZER_NAME,
            )

            # ── Ergebnis zusammenstellen ────────────────────────────
            result = {
                "training_size": n,
                "partition_index": k,
                "seed": seed,
                "model": config.MODEL_NAME,
                "optimizer": config.OPTIMIZER_NAME,
                **metrics,
            }

            all_results.append(result)
            completed += 1

            # ── Sofort speichern (Checkpoint) ───────────────────────
            save_results(all_results, results_path)

            print(
                f"[n={n:>5}, k={k}] "
                f"Test-Acc: {metrics['final_test_accuracy']:.4f} | "
                f"Val-Acc: {metrics['best_val_accuracy']:.4f} | "
                f"Epochen: {metrics['epochs_trained']} | "
                f"Zeit: {metrics['training_time_seconds']:.1f}s | "
                f"({completed}/{total_runs})"
            )

            # Speicher freigeben (KRITISCH!)
            del model, train_loader, val_loader, test_loader
            if config.DEVICE.type == "mps":
                torch.mps.empty_cache()
            import gc
            gc.collect()  # Python Garbage Collection erzwingen

    # ── Aggregation ─────────────────────────────────────────────────
    total_time = time.time() - experiment_start
    print(f"\nExperiment abgeschlossen in {total_time/60:.1f} Minuten.")

    aggregate_results(all_results, agg_path)


if __name__ == "__main__":
    run_experiment()
