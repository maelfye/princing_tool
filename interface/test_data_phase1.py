# interface/test_data_phase1.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_sources import get_market_data
from core.preprocessing import add_returns_and_vol

def main():
    df = get_market_data("AAPL", start="2024-01-01", end="2025-11-10")
    df = add_returns_and_vol(df, window=30)
    
    print("\n🔹 Aperçu des données :")
    print(df.head())
    print("\n📈 Statistiques :")
    print(df[["return", "vol_annual", "mu_annual"]].describe())

if __name__ == "__main__":
    main()


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_sources import get_multiple_assets

def main():
    tickers = ["AAPL", "MSFT", "NVDA"]
    data = get_multiple_assets(tickers, start="2024-01-01", end="2025-11-10")
    for t, df in data.items():
        print(f"\n🔹 {t}: {len(df)} lignes")
        print(df.head())

if __name__ == "__main__":
    main()


