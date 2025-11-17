import yfinance as yf

# Récupérer les données depuis le 1er janvier 2025 jusqu’à aujourd’hui
data = yf.download("AAPL", start="2025-01-01", end="2025-11-10", interval="1d")

# Manipuler directement les données en mémoire
print(data.head())
print(f"Nombre de lignes : {len(data)}")

# Exemple : calculer la variation journalière
data["daily_return"] = data["Close"].pct_change()
print(data.tail())