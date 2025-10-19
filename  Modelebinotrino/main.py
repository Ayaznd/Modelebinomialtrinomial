import yfinance as yf
import numpy as np

# Télécharger les prix de clôture de SPY sur 5 ans
spy_data = yf.download("SPY", period="5y")['Close']
prixSpy = spy_data.values
indices = np.linspace(0, len(prixSpy) - 1, 5, dtype=int)  # 15 points espacés
m = prixSpy[indices]

print(m)
m = np.array(m, dtype=float)

# Calcul des rendements logarithmiques
returns = np.log(m[1:] / m[:-1])

# Estimation de la variance à long terme (theta)
theta = np.var(returns, ddof=1)
print("Theta (variance à long terme) :", theta)

# Approximation des variances locales (instantanées)
variances = returns**2

# Estimation de sigma (volatilité de la variance)

sigma = np.std(returns, ddof=1)
print("Sigma (volatilité de la variance) :", sigma)