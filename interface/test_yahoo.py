# interface/test_yahoo.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_sources import get_yahoo_data

def main():
    ticker = "AAPL"  # ou autre : MSFT, TSLA, NVDA...
    df = get_yahoo_data(ticker, start="2025-01-01", end="2025-11-10")
    
    print("\nAperçu des données :")
    print(df.head())

    # Exemple de petit calcul : variation journalière
    df["daily_return"] = df["Close"].pct_change()
    print("\nDernières variations journalières :")
    print(df[["Date", "Close", "daily_return"]].tail())

if __name__ == "__main__":
    main()