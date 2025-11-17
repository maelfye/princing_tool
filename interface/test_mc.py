# interface/test_monte_carlo.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from core.monte_carlo import simulate_gbm

def value_at_risk(returns, confidence=0.95):
    """Calcule la Value at Risk (VaR) au niveau de confiance donné."""
    return np.percentile(returns, (1 - confidence) * 100)

def conditional_var(returns, confidence=0.95):
    """Calcule la Conditional VaR (Expected Shortfall)."""
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()

def main():
    # === Paramètres du modèle ===
    S0 = 230       # prix initial
    mu = 0.05      # drift annuel
    sigma = 0.2    # volatilité annuelle
    T = 1          # 1 an
    steps = 252
    n_paths = 50000
    seed = 42

    # === Simulation Monte Carlo ===
    paths = simulate_gbm(S0, mu, sigma, T, steps, n_paths, seed)
    ST = paths[-1]
    returns = (ST - S0) / S0   # rendement relatif

    # === Statistiques de base ===
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    var_95 = value_at_risk(returns, 0.95)
    cvar_95 = conditional_var(returns, 0.95)

    print("\n📊 --- Statistiques Monte Carlo ---")
    print(f"Rendement moyen      : {mean_ret*100:.2f}%")
    print(f"Volatilité empirique : {std_ret*100:.2f}%")
    print(f"VaR 95%              : {var_95*100:.2f}%")
    print(f"CVaR 95%             : {cvar_95*100:.2f}%")

    # === 1. Graphique des trajectoires ===
    plt.figure(figsize=(10, 5))
    plt.plot(paths[:, :50])  # 50 trajectoires
    plt.title(f"Monte Carlo - {n_paths} trajectoires GBM\nμ={mu}, σ={sigma}, T={T}")
    plt.xlabel("Jour")
    plt.ylabel("Prix simulé")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === 2. Histogramme du rendement (PnL) ===
    plt.figure(figsize=(9, 5))
    plt.hist(returns*100, bins=60, color='orange', edgecolor='white', alpha=0.7)
    plt.axvline(mean_ret*100, color='blue', linestyle='--', label=f"Moyenne = {mean_ret*100:.2f}%")
    plt.axvline(var_95*100, color='red', linestyle='--', label=f"VaR 95% = {var_95*100:.2f}%")
    plt.axvline(cvar_95*100, color='darkred', linestyle=':', label=f"CVaR 95% = {cvar_95*100:.2f}%")
    plt.title("Distribution des rendements simulés (PnL)")
    plt.xlabel("Rendement (%)")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()