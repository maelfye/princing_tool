# core/monte_carlo.py
import numpy as np

def simulate_gbm(S0: float, mu: float, sigma: float, T: float,
                 steps: int = 252, n_paths: int = 10000, seed: int = None) -> np.ndarray:
    """
    Simule des trajectoires de prix avec un mouvement brownien géométrique (GBM).
    
    Args:
        S0 (float): prix initial
        mu (float): rendement espéré (drift annuel)
        sigma (float): volatilité annuelle
        T (float): horizon en années (ex: 1 = 1 an)
        steps (int): nombre d'étapes temporelles (par défaut 252 jours ouvrés)
        n_paths (int): nombre de trajectoires simulées
        seed (int, optionnel): graine pour reproductibilité

    Returns:
        np.ndarray: matrice (steps, n_paths) contenant les prix simulés
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    Z = np.random.standard_normal((steps, n_paths))
    S = np.zeros_like(Z)
    S[0] = S0

    for t in range(1, steps):
        S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[t])

    return S

import numpy as np
import yfinance as yf

def simulate_asset_paths(tickers, T=1, steps=252, n_paths=1000, period="1y"):
    """
    Simule des trajectoires Monte Carlo pour plusieurs tickers
    en se basant sur leur drift et volatilité historiques.
    """
    dt = 1 / steps
    simulations = {}
    for ticker in tickers:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        data["return"] = data["Close"].pct_change()
        data.dropna(inplace=True)

        S0 = float(data["Close"].iloc[-1])
        mu = float(data["return"].mean() * 252)
        sigma = float(data["return"].std() * np.sqrt(252))

        # Simule n_paths trajectoires GBM
        Z = np.random.standard_normal((steps, n_paths))
        S = np.zeros_like(Z)
        S[0] = S0
        for t in range(1, steps):
            S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])
        simulations[ticker] = S
    return simulations