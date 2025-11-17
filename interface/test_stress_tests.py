# interface/test_stress_tests.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from core.stress_tests import stress_test_portfolio

def main():
    tickers = ["AAPL", "MSFT", "NVDA", "GOOG"]
    weights = [0.25, 0.25, 0.25, 0.25]

    result = stress_test_portfolio(
        tickers,
        weights,
        start="2024-01-01",
        end="2025-11-10",
        price_shock=-0.3,
        vol_factor=2.0
    )

    print("\n⚠️ --- Stress Test Results ---")
    print(result.round(4))

    # Visualisation simple
    result.T.plot(kind="bar", figsize=(9, 5), title="Risque avant/après choc")
    plt.ylabel("Valeur")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()