# core/preprocessing.py
import pandas as pd
import numpy as np

def add_returns_and_vol(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Calcule les rendements journaliers log, la volatilité annualisée et le drift.
    Args:
        df (pd.DataFrame): colonnes ['Date', 'Close']
        window (int): taille de la fenêtre mobile pour la volatilité
    Returns:
        pd.DataFrame: colonnes ['Date', 'Close', 'return', 'vol_annual', 'mu_annual']
    """
    df = df.copy()
    df["return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["vol_annual"] = df["return"].rolling(window).std() * np.sqrt(252)
    df["mu_annual"] = df["return"].rolling(window).mean() * 252
    df.dropna(inplace=True)
    return df

from core.preprocessing import add_returns_and_vol

def get_multiple_assets(tickers, start="2020-01-01", end="2025-11-10", preprocess=True):
    data = {}
    for t in tickers:
        try:
            df = get_market_data(t, start=start, end=end)
            if preprocess:
                df = add_returns_and_vol(df)
            data[t] = df
            print(f"✅ {t} téléchargé et pré-traité ({len(df)} lignes)")
        except Exception as e:
            print(f"⚠️ Erreur lors du téléchargement de {t}: {e}")
    return data