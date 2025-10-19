'''import numpy as np #justement pour tester sinn pas besoin
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from functions import *
n=100
nbj=88
strikes=[385,395,400,405,410,415,420]
s=[561.75,564.69,565.05,565.28,559.96,561.93,558.93]
tho=88/365
k=500
s0=542.9
r=0.042#4.2%
q=0.012
dt=1/252
sigma=0.195
T=88/365
a=[binomialcalleur(s0,T,k,sigma,r,n) for k,s0 in zip(strikes,s)]
print(np.array(a))
a=[trinomialcalleur(sigma,r,0.0,s0,T,k,n) for k,s0 in zip(strikes,s)]
print(np.array(a))
s=[532.17,532.17,546.77,534.65,531.65,544.22]
strikes=[150,155,300,305,360,400]
thoo=[253/365,253/365,238/365,252/365,247/365,231/365]

a=[binomialcalleur(s0,T,k,sigma,r,n) for k,s0,T in zip(strikes,s,thoo)]
print(np.array(a))
a=[trinomialcalleur(sigma,r,0.0,s0,T,k,n) for k,s0,T in zip(strikes,s,thoo)]
print(np.array(a))'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from functions import binomialcalleur, trinomialcalleur

# --- Paramètres de base ---
r = 0.042     # taux sans risque
sigma = 0.195 # volatilité
T = 88/365    # maturité
strikes = [385, 395, 400, 405, 410, 415, 420]
s_values = [561.75, 564.69, 565.05, 565.28, 559.96, 561.93, 558.93]

# --- Fonction Black-Scholes ---
def fonctrep(s0, k, r, sigma, T):
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def frCall(s0, k, r, T, sigma):
    d1, d2 = fonctrep(s0, k, r, sigma, T)
    return s0 * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)

# --- Étude de convergence ---
n_values = np.arange(1, 100, 5)
k = 400   # Strike choisi pour la démonstration
s0 = 565.05

bs_price = frCall(s0, k, r, T, sigma)
print(bs_price)
binomial_prices = [binomialcalleur(s0, T, k, sigma, r, n) for n in n_values]
trinomial_prices = [trinomialcalleur(sigma, r, 0.0, s0, T, k, n) for n in n_values]
import matplotlib.pyplot as plt
import numpy as np

# --- Tracé amélioré ---
plt.figure(figsize=(10, 6))

# Courbes principales
plt.plot(n_values, binomial_prices, 'b-', linewidth=2, label='Binomial', marker='o', markersize=3)
plt.plot(n_values, trinomial_prices, 'r-', linewidth=2, label='Trinomial', marker='s', markersize=3)
plt.axhline(y=bs_price, color='black', linestyle='--', linewidth=2, label='Black-Scholes (Analytique)')

# Amélioration des axes
plt.title("Convergence des modèles binomial et trinomial vers Black-Scholes", fontsize=14, fontweight='bold')
plt.xlabel("Nombre de périodes (n)", fontsize=12)
plt.ylabel("Prix de l'option Call", fontsize=12)

# Amélioration des ticks sur l'axe x
plt.xticks(n_values[::len(n_values)//10])  # Affiche environ 10 ticks répartis
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))  # Format entier

# Amélioration des ticks sur l'axe y
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))  # 2 décimales

# Grille et légende
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Ajustement des marges
plt.tight_layout()

# Afficher la valeur Black-Scholes sur le graphique
plt.text(0.02, 0.98, f'Black-Scholes = {bs_price:.4f}',
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.show()