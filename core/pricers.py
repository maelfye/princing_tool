# core/pricers.py
import numpy as np
from scipy.stats import norm
import yfinance as yf
from core.monte_carlo import simulate_gbm


# ========================
# ⚙️ 1. Black-Scholes model
# ========================
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ========================
# ⚙️ 2. Monte Carlo option pricing
# ========================
def monte_carlo_option_price(S0, K, T, r, sigma, option_type="call", n_paths=100_000, steps=252, seed=None):
    paths = simulate_gbm(S0, r, sigma, T, steps=steps, n_paths=n_paths, seed=seed)
    ST = paths[-1]
    if option_type.lower() == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    return np.exp(-r * T) * payoffs.mean()


# ========================
# ⚙️ 3. Market parameters
# ========================
def get_market_parameters(ticker: str, period: str = "1y"):
    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"Aucune donnée reçue pour {ticker}")

    data["return"] = data["Close"].pct_change()
    data.dropna(inplace=True)

    S0 = data["Close"].iloc[-1].item()
    sigma = data["return"].std() * np.sqrt(252)
    mu = data["return"].mean() * 252
    return S0, mu, sigma


# ========================
# ⚙️ 4. Risk-free rate
# ========================
def get_risk_free_rate(region: str = "US"):
    """
    Récupère automatiquement le taux sans risque selon la région.
    - US: ^TNX (10-Year Treasury)
    - EU: ^EURI10Y (Euro Area 10-Year Govt Bond Yield)
    - UK: ^GUKG10 (UK 10-Year Gilt)
    Retourne le taux en décimal (ex: 0.04 pour 4%)
    """
    tickers = {
        "US": "^TNX",
        "EU": "^EURI10Y",
        "UK": "^GUKG10",
    }

    try:
        symbol = tickers.get(region, "^TNX")
        data = yf.download(symbol, period="5d", auto_adjust=True, progress=False)
        latest = data["Close"].iloc[-1].item()
        return latest / 100  # ✅ Yahoo renvoie le taux en %, donc on le divise par 100
    except Exception:
        # Valeurs par défaut si Yahoo échoue
        default_rates = {"US": 0.04, "EU": 0.03, "UK": 0.045}
        return default_rates.get(region, 0.04)