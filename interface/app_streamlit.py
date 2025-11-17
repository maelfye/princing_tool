# interface/app_streamlit.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.monte_carlo import simulate_gbm
from core.risk_metrics import value_at_risk, conditional_var, max_drawdown, sharpe_ratio, simulate_portfolio_var
from core.portfolio import build_portfolio_returns, portfolio_statistics
from core.stress_tests import stress_test_portfolio
from core.pricers import black_scholes_price, monte_carlo_option_price, get_market_parameters, get_risk_free_rate
from core.multi_asset_simulation import simulate_correlated_assets
from core.basket_pricer import monte_carlo_basket_option

# === Configuration générale ===
st.set_page_config(page_title="Quant Dashboard", layout="wide")

st.title("💼 Quantitative Pricing & Risk Management Dashboard")

# --- Créer des onglets ---
tab1, tab2 = st.tabs(["📊 Risk Management", "💰 Pricing Options"])

# =========================================================
# 🟢 Onglet 1 : RISK MANAGEMENT
# =========================================================
with tab1:
    st.sidebar.header("⚙️ Paramètres du Risk Engine")

    tickers = st.sidebar.multiselect(
        "Tickers du portefeuille",
        ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA"],
        default=["AAPL", "MSFT", "NVDA"]
    )

    weights = []
    st.sidebar.markdown("### Pondérations (%)")
    for t in tickers:
        w = st.sidebar.slider(f"{t}", 0, 100, 100 // len(tickers))
        weights.append(w)
    weights = np.array(weights) / np.sum(weights)

    st.sidebar.markdown("### Simulation Monte Carlo")
    S0 = st.sidebar.number_input("Prix initial S₀", value=230.0)
    mu = st.sidebar.number_input("Drift annuel μ", value=0.05)
    sigma = st.sidebar.number_input("Volatilité annuelle σ", value=0.2)
    T = st.sidebar.number_input("Durée (années)", value=1.0)
    n_paths = st.sidebar.slider("Nb trajectoires", 1000, 50000, 10000, step=1000)

    st.sidebar.markdown("### Stress test")
    price_shock = st.sidebar.slider("Choc de prix (%)", -80, 0, -30) / 100
    vol_factor = st.sidebar.slider("Facteur de volatilité", 1.0, 3.0, 2.0)

    # ---- Simulation Monte Carlo simple ----
    st.subheader("📈 Simulation Monte Carlo")
    paths = simulate_gbm(S0, mu, sigma, T, steps=252, n_paths=n_paths, seed=42)
    ST = paths[-1]
    returns = (ST - S0) / S0

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(paths[:, :50], alpha=0.6)
    ax1.set_title(f"{len(paths[0])} trajectoires simulées (μ={mu}, σ={sigma})")
    ax1.set_xlabel("Jours")
    ax1.set_ylabel("Prix simulé")
    st.pyplot(fig1)

    # ---- Métriques ----
    var_95 = value_at_risk(returns, 0.95)
    cvar_95 = conditional_var(returns, 0.95)
    sharpe = sharpe_ratio(returns / 252)
    mdd = max_drawdown(pd.Series(returns))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VaR 95%", f"{var_95*100:.2f} %")
    col2.metric("CVaR 95%", f"{cvar_95*100:.2f} %")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Max Drawdown", f"{mdd*100:.2f} %")

    # ---- Histogramme ----
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(returns * 100, bins=60, color='orange', edgecolor='white', alpha=0.7)
    ax2.axvline(var_95 * 100, color='red', linestyle='--', label=f"VaR 95% = {var_95*100:.2f}%")
    ax2.axvline(cvar_95 * 100, color='darkred', linestyle=':', label=f"CVaR 95% = {cvar_95*100:.2f}%")
    ax2.set_title("Distribution des rendements simulés")
    ax2.set_xlabel("Rendement (%)")
    ax2.legend()
    st.pyplot(fig2)

    # ---- Stress Test ----
    if st.sidebar.button("Lancer le stress test réel"):
        result = stress_test_portfolio(
            tickers, weights,
            start="2024-01-01", end="2025-11-10",
            price_shock=price_shock, vol_factor=vol_factor
        )
        st.dataframe(result.round(4))
        st.bar_chart(result.T)

    # ---- VaR de portefeuille ----
    st.subheader("💼 Value-at-Risk (VaR) du portefeuille")

    alpha = st.select_slider("Niveau de confiance", options=[0.90, 0.95, 0.99], value=0.95)
    n_paths = st.slider("Nombre de simulations", 1000, 10000, 3000)

    if st.button("📉 Calculer la VaR simulée"):
        st.write("Simulation en cours...")
        st.write("Tickers :", tickers)
        st.write("Poids :", weights)
        result = simulate_portfolio_var(tickers, weights, n_paths=n_paths, alpha=alpha)
        var = result["VaR"]
        es = result["ES"]
        final_returns = result["final_returns"]

        st.success(f"✅ VaR {int(alpha*100)}% = {var*100:.2f} %")
        st.info(f"📉 Expected Shortfall (ES) = {es*100:.2f} %")

        fig, ax = plt.subplots()
        ax.hist(final_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(var, color='red', linestyle='--', label=f'VaR {int(alpha*100)}%')
        ax.axvline(es, color='darkred', linestyle='--', label=f'ES {int(alpha*100)}%')
        ax.set_title("Distribution simulée des rendements du portefeuille")
        ax.legend()
        st.pyplot(fig)

    # ---- Simulation corrélée ----
    st.subheader("📊 Simulation Monte Carlo corrélée")
    if st.button("🎲 Lancer la simulation corrélée"):
        simulations, corr_matrix, mus, sigmas = simulate_correlated_assets(
            tickers, T=T, steps=252, n_paths=1000, period="1y"
        )
        fig, ax = plt.subplots()
        for ticker in tickers:
            ax.plot(simulations[ticker][:, :10])
        ax.set_title(f"Simulation Monte Carlo corrélée ({', '.join(tickers)})")
        st.pyplot(fig)
        st.subheader("📈 Matrice de corrélation empirique")
        st.dataframe(np.round(corr_matrix, 2))

# =========================================================
# 🟡 Onglet 2 : PRICING OPTIONS
# =========================================================
with tab2:
    st.header("💰 Pricing automatique d'options européennes")

    # ================================
    # ⚙️ Option mono-actif
    # ================================
    option_type = st.selectbox("Type d’option", ["call", "put"])
    ticker = st.text_input("Ticker de l’actif (ex: AAPL, MSFT, NVDA)", value="AAPL")
    T = st.number_input("Maturité (en années)", value=1.0)
    K = st.number_input("Strike (prix d’exercice)", value=220.0)
    n_paths = st.slider("Nombre de simulations Monte Carlo", 1000, 100000, 10000, step=1000)

    if ticker:
        try:
            S, mu, sigma = get_market_parameters(ticker)
            market = st.selectbox("🌍 Marché de référence", ["US", "EU", "UK"], index=0)
            r = get_risk_free_rate(market)
            st.write(f"**Taux sans risque ({market})** : {r*100:.2f} %")

            st.success(f"✅ Données récupérées automatiquement pour {ticker}")
            st.write(f"**Prix actuel (S₀)** : {S:.2f} $")
            st.write(f"**Drift annuel (μ)** : {mu*100:.2f} %")
            st.write(f"**Volatilité annuelle (σ)** : {sigma*100:.2f} %")

            bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
            mc_price = monte_carlo_option_price(S, K, T, r, sigma, option_type, n_paths=n_paths)

            st.metric("Prix Black-Scholes", f"{bs_price:.2f} $")
            st.metric("Prix Monte Carlo", f"{mc_price:.2f} $")
            st.write(f"Différence (MC - BS) : `{(mc_price - bs_price):.4f}`")

            # Payoff
            st.subheader("💹 Payoff à maturité")
            S_range = np.linspace(0.5*K, 1.5*K, 200)
            payoff = np.maximum(S_range - K, 0) if option_type == "call" else np.maximum(K - S_range, 0)
            fig, ax = plt.subplots()
            ax.plot(S_range, payoff, color="green")
            ax.set_title(f"Payoff de l’option {option_type.upper()} (K={K})")
            ax.set_xlabel("Prix final S_T")
            ax.set_ylabel("Payoff")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors de la récupération : {e}")

    # ================================
    # 🧺 Basket Option Pricing
    # ================================
    st.markdown("---")
    st.subheader("🧺 Pricing d'une Basket Option (multi-actifs)")

    basket_tickers = st.multiselect(
        "Choisis les actifs du panier",
        ["AAPL", "MSFT", "NVDA", "META", "TSLA", "GOOG"],
        default=["AAPL", "MSFT", "NVDA"]
    )

    basket_weights = []
    for t in basket_tickers:
        w = st.slider(f"Pondération de {t}", 0.0, 1.0, 1.0 / len(basket_tickers))
        basket_weights.append(w)
    basket_weights = np.array(basket_weights) / np.sum(basket_weights)

    basket_K = st.number_input("Strike du panier (K)", value=250.0)
    basket_T = st.number_input("Maturité du panier (années)", value=1.0)
    basket_type = st.selectbox("Type d’option basket", ["call", "put"])
    basket_r = st.selectbox("🌍 Marché de référence (basket)", ["US", "EU", "UK"], index=0)
    r_value = get_risk_free_rate(basket_r)
    st.write(f"Taux sans risque : {r_value*100:.2f} %")

    # ===== Corrélation : choix empirique ou personnalisée =====
    corr_mode = st.radio(
        "Choix du type de corrélation :",
        ["Empirique (marché réel)", "Personnalisée (entrée manuelle)"]
    )
    use_custom_corr = (corr_mode == "Personnalisée (entrée manuelle)")

    if use_custom_corr:
        st.markdown("✏️ **Définis ta matrice de corrélation (symétrique entre -1 et 1)**")
        default_corr = pd.DataFrame(np.identity(len(basket_tickers)), 
                                    index=basket_tickers, columns=basket_tickers)
        user_corr_df = st.data_editor(default_corr, num_rows="fixed")
    else:
        user_corr_df = None

    # ===== Calcul du prix =====
    if st.button("💰 Calculer le prix de la Basket Option"):
        with st.spinner("Simulation Monte Carlo en cours..."):
            price, mus, sigmas, corr = monte_carlo_basket_option(
                basket_tickers,
                basket_weights,
                basket_K,
                basket_T,
                r_value,
                basket_type,
                n_paths=3000,
                custom_corr=user_corr_df.to_numpy() if use_custom_corr else None
            )

        st.success(f"✅ Prix de la Basket Option ({basket_type.upper()}) : **{price:.2f} $**")

        # Affichage propre
        df_summary = pd.DataFrame({
            "Actif": mus.index,
            "Drift annuel (μ)": (mus.values * 100).round(2),
            "Volatilité annuelle (σ)": (sigmas.values * 100).round(2)
        })

        st.subheader("📊 Paramètres de marché du panier")
        st.dataframe(df_summary.style.format({
            "Drift annuel (μ)": "{:.2f} %",
            "Volatilité annuelle (σ)": "{:.2f} %"
        }))

        # Heatmap de corrélation
        corr_df = pd.DataFrame(np.round(corr, 2), index=mus.index, columns=mus.index)
        st.subheader("📈 Corrélation entre actifs du panier")
        st.dataframe(corr_df.style.background_gradient(cmap="coolwarm"))