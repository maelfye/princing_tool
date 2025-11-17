# core/data_sources.py
import yfinance as yf
import pandas as pd
import numpy as np

def get_market_data(ticker: str, start: str = "2020-01-01", end: str = "2025-11-10") -> pd.DataFrame:
    """
    Télécharge les prix de clôture journaliers d'un actif depuis Yahoo Finance.

    Args:
        ticker (str): symbole du titre (ex: 'AAPL', 'MSFT', 'SPY')
        start (str): date de début (format 'YYYY-MM-DD')
        end (str): date de fin (format 'YYYY-MM-DD')

    Returns:
        pd.DataFrame: colonnes ['Date', 'Close', 'ticker']
    """
    print(f"📡 Téléchargement des données pour {ticker}...")
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)

    if "Close" not in df.columns or df.empty:
        raise ValueError(f"⚠️ Données invalides ou vides pour {ticker}")

    df = df[["Close"]].copy()
    df.reset_index(inplace=True)
    df["ticker"] = ticker
    print(f"✅ {len(df)} observations téléchargées pour {ticker}")
    return df

def get_multiple_assets(tickers, start="2020-01-01", end="2025-11-10"):
    """
    Télécharge plusieurs tickers et renvoie un dictionnaire {ticker: DataFrame}.
    Chaque DataFrame contient ['Date', 'Close', 'ticker'].
    """
    data = {}
    for t in tickers:
        try:
            df = get_market_data(t, start=start, end=end)
            data[t] = df
            print(f"✅ {t} ajouté ({len(df)} lignes)")
        except Exception as e:
            print(f"⚠️ Erreur lors du téléchargement de {t}: {e}")
    return data