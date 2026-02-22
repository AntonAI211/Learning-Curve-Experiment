"""
Visualisierung: Lernkurve mit Power-Law-Fit, Residuenplot,
und optionale Trainingskurven.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from fit_power_law import fit_power_law, power_law


# Matplotlib-Einstellungen für saubere Plots
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def plot_learning_curve(fit_result: dict, save_path: str = None):
    """
    HAUPTPLOT: Lernkurve mit Power-Law-Fit.
    x-Achse: n (log), y-Achse: Accuracy
    """
    n_values = np.array(fit_result["n_values"])
    observed = np.array(fit_result["observed"])
    std_vals = np.array(fit_result["std_values"])
    A_inf = fit_result["A_inf"]
    eta = fit_result["eta"]
    gamma = fit_result["gamma"]
    r_sq = fit_result["r_squared"]

    # Feine Kurve für den Fit
    n_fine = np.logspace(np.log10(n_values.min()), np.log10(n_values.max()), 500)
    a_fine = power_law(n_fine, A_inf, eta, gamma)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Datenpunkte mit Fehlerbalken
    ax.errorbar(
        n_values, observed, yerr=std_vals,
        fmt="o", color="#2196F3", markersize=8, capsize=5, capthick=1.5,
        elinewidth=1.5, label="Gemessene Accuracy (Mittelwert)", zorder=3,
    )

    # Gefittete Kurve
    ax.plot(
        n_fine, a_fine, "-", color="#F44336", linewidth=2.5,
        label=(
            f"Power-Law-Fit: $A(n) = {A_inf:.4f} + ({eta:.4f}) \\cdot n^{{{gamma:.4f}}}$\n"
            f"$R^2 = {r_sq:.4f}$"
        ),
        zorder=2,
    )

    # Asymptote
    ax.axhline(
        y=A_inf, color="#9E9E9E", linestyle="--", linewidth=1.5,
        label=f"Asymptote $A_\\infty = {A_inf:.4f}$", zorder=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Trainingsdatenmenge $n$")
    ax.set_ylabel("Accuracy $A(n)$")
    ax.set_title("Lernkurve: $A(n) = A_\\infty + \\eta \\cdot n^\\gamma$")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0, 1.05)

    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "lernkurve_power_law_fit.png")
    fig.savefig(save_path)
    print(f"Plot gespeichert: {save_path}")
    plt.close(fig)


def plot_residuals(fit_result: dict, save_path: str = None):
    """
    RESIDUENPLOT: Abweichung zwischen Modell und Daten.
    """
    n_values = np.array(fit_result["n_values"])
    residuals = np.array(fit_result["residuals"])

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        range(len(n_values)), residuals, color="#4CAF50", alpha=0.8,
        tick_label=[str(int(n)) for n in n_values],
    )
    ax.axhline(y=0, color="black", linewidth=0.8)

    ax.set_xlabel("Trainingsdatenmenge $n$")
    ax.set_ylabel("Residuum ($A_{beob.} - A_{pred.}$)")
    ax.set_title("Residuen des Power-Law-Fits")
    ax.grid(True, alpha=0.3, axis="y")

    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "residuenplot.png")
    fig.savefig(save_path)
    print(f"Plot gespeichert: {save_path}")
    plt.close(fig)


def plot_training_curves(save_path: str = None):
    """
    Optionaler Plot: Trainingskurven (Loss/Accuracy pro Epoche) für
    verschiedene n, um zu zeigen, dass kleinere n schneller konvergieren.
    Zeigt jeweils die erste Partition (k=0).
    """
    results_path = os.path.join(config.RESULTS_DIR, config.INDIVIDUAL_RESULTS_FILE)
    if not os.path.exists(results_path):
        print("Keine individuellen Ergebnisse gefunden.")
        return

    with open(results_path, "r") as f:
        all_results = json.load(f)

    # Wähle k=0 für ausgewählte n-Werte
    selected_n = [100, 500, 2500, 10000, 48000]
    colors = ["#F44336", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for n_val, color in zip(selected_n, colors):
        for r in all_results:
            if r["training_size"] == n_val and r["partition_index"] == 0:
                epochs = range(1, len(r["val_accuracy_per_epoch"]) + 1)
                ax1.plot(
                    epochs, r["val_accuracy_per_epoch"],
                    color=color, label=f"n = {n_val:,}", linewidth=1.5,
                )
                ax2.plot(
                    epochs, r["val_loss_per_epoch"],
                    color=color, label=f"n = {n_val:,}", linewidth=1.5,
                )
                break

    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_title("Validation Accuracy pro Epoche")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoche")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss pro Epoche")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Trainingskurven für verschiedene Trainingsgrößen", fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "trainingskurven.png")
    fig.savefig(save_path)
    print(f"Plot gespeichert: {save_path}")
    plt.close(fig)


def plot_derivative(fit_result: dict, save_path: str = None):
    """
    Zusatzplot: Erste Ableitung A'(n) = eta * gamma * n^(gamma-1)
    Visualisiert die abnehmenden Grenzerträge.
    """
    A_inf = fit_result["A_inf"]
    eta = fit_result["eta"]
    gamma = fit_result["gamma"]

    n_fine = np.logspace(
        np.log10(min(fit_result["n_values"])),
        np.log10(max(fit_result["n_values"])),
        500,
    )
    # Erste Ableitung
    dA_dn = eta * gamma * np.power(n_fine, gamma - 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(n_fine, dA_dn, color="#FF5722", linewidth=2.5)
    ax.set_xscale("log")
    ax.set_xlabel("Trainingsdatenmenge $n$")
    ax.set_ylabel("$A'(n) = \\eta \\cdot \\gamma \\cdot n^{(\\gamma - 1)}$")
    ax.set_title("Grenzertrag: Erste Ableitung der Lernkurve")
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(y=0, color="black", linewidth=0.5)

    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "grenzertrag_ableitung.png")
    fig.savefig(save_path)
    print(f"Plot gespeichert: {save_path}")
    plt.close(fig)


def create_all_plots():
    """Erstellt alle Plots."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print("Starte Power-Law-Fit...")
    fit_result = fit_power_law()

    print("\nErstelle Plots...")
    plot_learning_curve(fit_result)
    plot_residuals(fit_result)
    plot_training_curves()
    plot_derivative(fit_result)

    print("\nAlle Plots erstellt.")


if __name__ == "__main__":
    create_all_plots()
