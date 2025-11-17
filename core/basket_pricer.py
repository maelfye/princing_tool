import numpy as np
import pandas as pd
import yfinance as yf

def monte_carlo_basket_option(
    tickers, 
    weights, 
    K, 
    T, 
    r, 
    option_type="call", 
    n_paths=10000, 
    custom_corr=None
):
    """
    Pricing Monte Carlo d'une option panier (basket option).
    - tickers : liste d'actifs (ex: ["AAPL", "MSFT", "NVDA"])
    - weights : pondérations du panier (somme = 1)
    - K : strike
    - T : maturité (en années)
    - r : taux sans risque
    - option_type : "call" ou "put"
    - n_paths : nombre de simulations Monte Carlo
    - custom_corr : matrice de corrélation personnalisée (optionnelle)
    """

    # ==========================
    # 1️⃣ Téléchargement des données
    # ==========================
    # ==========================
    raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False)

    # Gérer le format selon le nombre de tickers
    if isinstance(raw.columns, pd.MultiIndex):
        # Plusieurs tickers → prendre la colonne 'Close' (après auto_adjust, c’est déjà ajusté)
        if "Close" in raw.columns.get_level_values(0):
            data = raw["Close"]
        else:
            raise KeyError("Colonne 'Close' introuvable dans les données téléchargées.")
    else:
        # Un seul ticker → forcer le format DataFrame
        if "Close" in raw.columns:
            data = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise KeyError("Colonne 'Close' introuvable pour le ticker unique.")

    data = data.dropna()

    # ==========================
    # 2️⃣ Calcul des paramètres de marché
    # ==========================
    log_returns = np.log(data / data.shift(1)).dropna()
    mus = log_returns.mean() * 252
    sigmas = log_returns.std() * np.sqrt(252)

    # ==========================
    # 3️⃣ Corrélation
    # ==========================
    if custom_corr is not None:
        corr = np.array(custom_corr)
    else:
        corr = log_returns.corr().to_numpy()

    # Vérification que la matrice est semi-définie positive
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        raise ValueError("La matrice de corrélation n’est pas définie positive.")

    # ==========================
    # 4️⃣ Simulation Monte Carlo
    # ==========================
    S0 = data.iloc[-1].values
    n_assets = len(tickers)
    dt = 1 / 252

    payoffs = np.zeros(n_paths)

    for i in range(n_paths):
        Z = np.random.normal(size=n_assets)
        correlated_Z = L @ Z

        # Simulation de prix à maturité (log-normal)
        ST = S0 * np.exp((mus - 0.5 * sigmas**2) * T + sigmas * np.sqrt(T) * correlated_Z)

        # Prix du panier
        basket_price = np.dot(weights, ST)

        # Payoff selon le type d’option
        if option_type.lower() == "call":
            payoff = max(basket_price - K, 0)
        else:
            payoff = max(K - basket_price, 0)

        payoffs[i] = payoff

    # ==========================
    # 5️⃣ Prix de l’option
    # ==========================
    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price, mus, sigmas, corr