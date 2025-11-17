# interface/test_risk_metrics.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.monte_carlo import simulate_gbm
from core.risk_metrics import value_at_risk, conditional_var, max_drawdown, sharpe_ratio

def main():
    # --- paramètres de simulation ---
    S0 = 230
    mu = 0.05
    sigma = 0.2
    T = 1
    steps = 252
    n_paths = 50000
    seed = 42

    # --- simulation Monte Carlo ---
    paths = simulate_gbm(S0, mu, sigma, T, steps, n_paths, seed)
    ST = paths[-1]
    returns = (ST - S0) / S0  # rendement sur la période

    # --- calcul des métriques de risque ---
    var_95 = value_at_risk(returns, 0.95)
    cvar_95 = conditional_var(returns, 0.95)
    sharpe = sharpe_ratio(returns / steps)  # approx rendements journaliers
    # pour max_drawdown, on simule une série de rendements journaliers fictive
    # équivalente à un scénario moyen :
    returns_daily = np.diff(paths.mean(axis=1)) / paths.mean(axis=1)[:-1]
    returns_daily = pd.Series(returns_daily)
    mdd = max_drawdown(returns_daily)

    print("\n📊 --- RISK METRICS ---")
    print(f"VaR 95%     : {var_95*100:.2f} %")
    print(f"CVaR 95%    : {cvar_95*100:.2f} %")
    print(f"Sharpe (ann): {sharpe:.2f}")
    print(f"Max Drawdown: {mdd*100:.2f} %")

    # --- visualisation de la distribution des rendements ---
    plt.figure(figsize=(9, 5))
    plt.hist(returns*100, bins=60, edgecolor='white', alpha=0.7)
    plt.axvline(var_95*100, linestyle='--', label=f"VaR 95% = {var_95*100:.2f}%")
    plt.axvline(cvar_95*100, linestyle=':', label=f"CVaR 95% = {cvar_95*100:.2f}%")
    plt.title("Distribution des rendements simulés (PnL)")
    plt.xlabel("Rendement (%)")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()