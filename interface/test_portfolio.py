# interface/test_portfolio.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from core.portfolio import build_portfolio_returns, portfolio_statistics

def main():
    tickers = ["AAPL", "MSFT", "NVDA", "GOOG"]
    weights = [0.25, 0.25, 0.25, 0.25]  # égal pondéré

    returns_df, weights = build_portfolio_returns(tickers, weights, start="2024-01-01", end="2025-11-10")
    stats = portfolio_statistics(returns_df, weights)

    print("\n📊 --- Portefeuille multi-actifs ---")
    print(f"Actifs : {tickers}")
    print(f"Pondérations : {weights}")
    print("\n--- Statistiques ---")
    for k, v in stats.items():
        if 'VaR' in k or 'CVaR' in k:
            print(f"{k:20s}: {v*100:.2f} %")
        else:
            print(f"{k:20s}: {v:.4f}")

    # === visualisation ===
    plt.figure(figsize=(10, 5))
    plt.plot((1 + returns_df["portfolio_return"]).cumprod(), label="Portefeuille")
    plt.title("Évolution du portefeuille (rendements cumulés)")
    plt.xlabel("Temps")
    plt.ylabel("Cumul rendement")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()