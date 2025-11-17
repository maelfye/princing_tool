# core/portfolio.py
import numpy as np
import pandas as pd
from core.data_sources import get_multiple_assets
from core.preprocessing import add_returns_and_vol
from core.risk_metrics import value_at_risk, conditional_var

def build_portfolio_returns(tickers, weights, start="2020-01-01", end="2025-11-10"):
    """
    Construit une matrice de rendements pour plusieurs actifs et calcule
    le rendement agrégé du portefeuille.
    """
    data = get_multiple_assets(tickers, start=start, end=end)
    merged = pd.DataFrame()

    for t in tickers:
        df = data[t]
        df = add_returns_and_vol(df)
        merged[t] = df["return"].reset_index(drop=True)

    merged.dropna(inplace=True)
    weights = np.array(weights) / np.sum(weights)  # normalisation

    # rendements journaliers du portefeuille
    merged["portfolio_return"] = merged.dot(weights)
    return merged, weights


def portfolio_statistics(returns_df, weights):
    """
    Calcule les stats de risque du portefeuille : vol, VaR, CVaR.
    """
    portfolio_returns = returns_df["portfolio_return"]

    # matrice de covariance
    cov_matrix = returns_df.iloc[:, :-1].cov().values
    vol_portfolio = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)

    # risk metrics
    var_95 = value_at_risk(portfolio_returns, 0.95)
    cvar_95 = conditional_var(portfolio_returns, 0.95)

    stats = {
        "volatility_annual": vol_portfolio,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "mean_daily_return": portfolio_returns.mean(),
    }
    return stats