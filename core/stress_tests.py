# core/stress_tests.py
import numpy as np
import pandas as pd
from core.portfolio import build_portfolio_returns, portfolio_statistics

def apply_price_shock(df: pd.DataFrame, shock: float = -0.3):
    """
    Applique un choc de marché sur les prix (ex: -0.3 = -30 %).
    """
    shocked_df = df.copy()
    shocked_df["Close"] *= (1 + shock)
    shocked_df["return"] = np.log(shocked_df["Close"] / shocked_df["Close"].shift(1))
    shocked_df.dropna(inplace=True)
    return shocked_df

def apply_volatility_shock(vol: float, factor: float = 2.0):
    """
    Applique un choc de volatilité (ex: factor=2 => volatilité doublée).
    """
    return vol * factor

def stress_test_portfolio(tickers, weights, start, end,
                          price_shock=-0.3, vol_factor=2.0):
    """
    Compare les métriques de risque avant et après un choc de marché.
    """
    # Données de base
    base_returns, weights = build_portfolio_returns(tickers, weights, start, end)
    base_stats = portfolio_statistics(base_returns, weights)

    # Choc de prix (baisse de 30 %)
    shocked_returns = base_returns.copy()
    shocked_returns.iloc[:, :-1] = shocked_returns.iloc[:, :-1] * (1 + price_shock)
    shocked_returns["portfolio_return"] = shocked_returns.iloc[:, :-1].dot(weights)
    shocked_stats = portfolio_statistics(shocked_returns, weights)

    # Résumé
    comparison = pd.DataFrame({
        "Normal": base_stats,
        "Stress": shocked_stats
    })
    return comparison

def scenario_custom(df, price_shock=-0.2, vol_factor=1.5, correlation_boost=1.2):
    # ex : augmenter corrélation entre actifs
    cov = df.cov()
    boosted_cov = cov * correlation_boost
    return boosted_cov