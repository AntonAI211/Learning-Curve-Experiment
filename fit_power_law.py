"""
Power-Law-Fit: A(n) = A_inf + eta * n^gamma
Verwendet scipy.optimize.curve_fit.
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import config


def power_law(n, A_inf, eta, gamma):
    """Power-Law Modell: A(n) = A_inf + eta * n^gamma"""
    return A_inf + eta * np.power(n, gamma)


def fit_power_law(csv_path: str = None) -> dict:
    """
    Liest aggregierte Ergebnisse und fittet das Power-Law-Modell.
    Gibt die gefitteten Parameter und Gütemaße zurück.
    """
    if csv_path is None:
        csv_path = os.path.join(config.RESULTS_DIR, config.AGGREGATED_RESULTS_FILE)

    df = pd.read_csv(csv_path)
    df = df[df["n"] >= config.FIT_MIN_N].reset_index(drop=True)
    n_values = df["n"].values.astype(float)
    acc_values = df["mean_accuracy"].values
    std_values = df["std_accuracy"].values

    # Gewichtung: 1/sigma (Runs mit kleiner Streuung stärker gewichten)
    # Für n=48000 (std=0) setze kleinen Wert
    sigma = np.where(std_values > 0, std_values, 0.001)

    # ── Curve Fit ───────────────────────────────────────────────────
    try:
        popt, pcov = curve_fit(
            power_law,
            n_values,
            acc_values,
            p0=config.FIT_P0,
            bounds=(config.FIT_BOUNDS[0], config.FIT_BOUNDS[1]),
            sigma=sigma,
            absolute_sigma=True,
            maxfev=10000,
        )
    except RuntimeError as e:
        print(f"Curve Fit fehlgeschlagen: {e}")
        print("Versuche ohne Gewichtung...")
        popt, pcov = curve_fit(
            power_law,
            n_values,
            acc_values,
            p0=config.FIT_P0,
            bounds=(config.FIT_BOUNDS[0], config.FIT_BOUNDS[1]),
            maxfev=10000,
        )

    A_inf, eta, gamma = popt
    perr = np.sqrt(np.diag(pcov))  # Standardfehler der Parameter

    # ── R² berechnen ────────────────────────────────────────────────
    predicted = power_law(n_values, *popt)
    ss_res = np.sum((acc_values - predicted) ** 2)
    ss_tot = np.sum((acc_values - np.mean(acc_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # ── Residuen ────────────────────────────────────────────────────
    residuals = acc_values - predicted

    result = {
        "A_inf": A_inf,
        "eta": eta,
        "gamma": gamma,
        "A_inf_se": perr[0],
        "eta_se": perr[1],
        "gamma_se": perr[2],
        "r_squared": r_squared,
        "n_values": n_values.tolist(),
        "observed": acc_values.tolist(),
        "predicted": predicted.tolist(),
        "residuals": residuals.tolist(),
        "std_values": std_values.tolist(),
    }

    # ── Ausgabe ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Power-Law-Fit: A(n) = A_inf + eta * n^gamma")
    print("=" * 60)
    print(f"  A_inf (Asymptote)  = {A_inf:.6f}  (± {perr[0]:.6f})")
    print(f"  eta                = {eta:.6f}  (± {perr[1]:.6f})")
    print(f"  gamma              = {gamma:.6f}  (± {perr[2]:.6f})")
    print(f"  R²                 = {r_squared:.6f}")
    print("=" * 60)
    print(f"\n  → Maximale Genauigkeit:  {A_inf:.2%}")
    print(f"  → Sättigungsexponent:    {gamma:.4f}")
    print()

    # Tabelle: n, beobachtet, vorhergesagt, Residuum
    print(f"  {'n':>8}  {'A(n) beob.':>10}  {'A(n) pred.':>10}  {'Residuum':>10}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
    for i, n in enumerate(n_values):
        print(
            f"  {int(n):>8}  {acc_values[i]:>10.4f}  "
            f"{predicted[i]:>10.4f}  {residuals[i]:>+10.4f}"
        )

    return result


if __name__ == "__main__":
    fit_power_law()
