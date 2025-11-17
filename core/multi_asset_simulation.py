import numpy as np
import yfinance as yf

def simulate_correlated_assets(tickers, T=1, steps=252, n_paths=5000, period="1y"):
    """
    Simule des trajectoires Monte Carlo corrélées pour plusieurs actifs.
    Retourne un dictionnaire {ticker: matrice trajectoires}.
    """
    dt = 1 / steps

    # 1️⃣ Téléchargement des prix historiques
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)["Close"]
    returns = data.pct_change().dropna()

    # 2️⃣ Calcul des paramètres individuels
    mus = returns.mean() * 252
    sigmas = returns.std() * np.sqrt(252)
    S0s = data.iloc[-1].values

    # 3️⃣ Matrice de corrélation empirique
    corr_matrix = returns.corr().values

    # 4️⃣ Décomposition de Cholesky pour créer des chocs corrélés
    L = np.linalg.cholesky(corr_matrix)

    n_assets = len(tickers)
    simulations = {ticker: np.zeros((steps, n_paths)) for ticker in tickers}

    for p in range(n_paths):
        # Génération des chocs aléatoires corrélés
        Z = np.random.standard_normal((steps, n_assets))
        correlated_Z = Z @ L.T

        S = np.zeros((steps, n_assets))
        S[0, :] = S0s

        for t in range(1, steps):
            drift_term = (mus - 0.5 * sigmas**2) * dt
            diffusion = sigmas.values * np.sqrt(dt) * correlated_Z[t, :]
            S[t, :] = S[t-1, :] * np.exp(drift_term + diffusion)

        # Stocker chaque trajectoire
        for i, ticker in enumerate(tickers):
            simulations[ticker][:, p] = S[:, i]

    return simulations, corr_matrix, mus, sigmas