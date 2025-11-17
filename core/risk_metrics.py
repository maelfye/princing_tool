# core/risk_metrics.py
import numpy as np
import pandas as pd
import yfinance as yf

def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calcule la Value at Risk (VaR) pour une distribution de rendements.

    Args:
        returns (np.ndarray): rendements (ex: PnL relatif, ex: -0.05 = -5%)
        confidence (float): niveau de confiance (ex: 0.95 pour 95%)

    Returns:
        float: VaR (valeur négative = perte)
    """
    returns = np.asarray(returns)
    return np.percentile(returns, (1 - confidence) * 100)


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calcule la Conditional Value at Risk (CVaR), ou Expected Shortfall.

    Args:
        returns (np.ndarray): rendements
        confidence (float): niveau de confiance

    Returns:
        float: CVaR (perte moyenne au-delà de la VaR)
    """
    returns = np.asarray(returns)
    var = value_at_risk(returns, confidence)
    tail_losses = returns[returns <= var]
    if tail_losses.size == 0:
        return var
    return tail_losses.mean()


def max_drawdown(returns: pd.Series) -> float:
    """
    Calcule le maximum drawdown à partir d'une série de rendements journaliers.

    Args:
        returns (pd.Series): rendements journaliers (ex: -0.02 = -2%)

    Returns:
        float: drawdown max (valeur négative, ex: -0.35 = -35%)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0) -> float:
    """
    Calcule le Sharpe ratio annualisé à partir de rendements journaliers.

    Args:
        returns (np.ndarray): rendements journaliers
        rf (float): taux sans risque annuel (ex: 0.02 pour 2%)

    Returns:
        float: Sharpe ratio annualisé
    """
    returns = np.asarray(returns)
    if returns.size == 0:
        return np.nan

    # rendement excédentaire quotidien
    daily_rf = rf / 252
    excess = returns - daily_rf

    mu = excess.mean()
    sigma = excess.std()
    if sigma == 0:
        return np.nan

    sharpe_daily = mu / sigma
    sharpe_annual = sharpe_daily * np.sqrt(252)
    return float(sharpe_annual)



def simulate_portfolio_var(
    tickers, weights, n_paths=5000, alpha=0.95, T=1.0, period="1y"
):
    """
    VaR/ES Monte Carlo vectorisés à partir des stats historiques (mu, cov) des log-returns.
    - Aucun appel réseau pendant la simu, un seul download pour estimer mu/cov/S0.
    - Simule directement le log-return multi-actif sur l’horizon T (MVN).
    """
    tickers = list(tickers)
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    # 1) Données -> mu, cov (quotidien) + S0
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].dropna()
    else:
        # un seul ticker
        px = raw[["Close"]].rename(columns={"Close": tickers[0]}).dropna()

    log_ret = np.log(px/px.shift(1)).dropna()
    mu_d = log_ret.mean().values            # moyennes quotidiennes (vector)
    cov_d = log_ret.cov().values            # covariance quotidienne (matrix)
    S0 = px.iloc[-1].values                 # derniers prix (vector)

    # 2) Paramètres sur l’horizon T (≈ 252 * T jours)
    d = int(round(252 * T))
    m_T = mu_d * d
    C_T = cov_d * d

    # 3) Simulation MVN vectorisée (n_paths x n_assets)
    L = np.linalg.cholesky(C_T + 1e-12*np.eye(len(tickers)))
    Z = np.random.normal(size=(n_paths, len(tickers)))
    X = Z @ L.T + m_T  # log-returns sur l’horizon T

    # 4) Prix à maturité et rendement du portefeuille
    ST = S0 * np.exp(X)                     # (n_paths, n_assets)
    V0 = float(S0 @ w)
    VT = ST @ w
    final_returns = VT / V0 - 1.0           # (n_paths,)

    # 5) VaR (perte positive) et ES
    q = np.quantile(final_returns, 1 - alpha)
    var = -q
    es = -final_returns[final_returns <= q].mean()

    return {
        "VaR": float(var),
        "ES": float(es),
        "final_returns": final_returns,
        "mu_daily": mu_d,
        "cov_daily": C_T / d,
        "S0": S0,
        "weights": w,
        "alpha": alpha,
    }