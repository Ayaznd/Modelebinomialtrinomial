import numpy as np #justement pour tester sinn pas besoin
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
def binomialcalleur(s0,T,k,sigma,r,n):
    dt=T/n
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u
    R=np.exp(r*dt)
    p=(R-d)/(u-d)
    q=1-p
    ind=np.arange(n+1) #[i for i in range(n+1)] m'a generer probleme ??
    #Dans les options eur pas besoin de tracabilite
    sech=s0*u**ind*d**(n-ind) #prix du ss jacent a l'echeance
    call=np.maximum(sech-k,0) #les payoffs a l'echeance
    for i in range(n-1,-1,-1) :
        call=[(p*call[j+1]+q*call[j])/R for j in range(i+1)]
    return call[0]
def binomialputeur(s0,T,k,sigma,r,n):
    dt=T/n
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u
    R=np.exp(r*dt)
    p=(R-d)/(u-d)
    q=1-p
    ind=np.arange(n+1)
    #dans les options eur pas besoin de tracabilite
    sech=s0*u**ind*d**(n-ind) #prix du ss jacent a l'echeance
    put=np.maximum(k-sech,0) #les payoffs a l'echeance
    for i in range(n-1,-1,-1) :
        put=[(p*put[j+1]+q*put[j])/R for j in range(i+1)]
    return put[0]
#a partir de cette etape on est besoin de la tracabilite pour comparer l'immediat et l'attente
#on garde dans le ecteur le max entre eux
def binomialcallam(s0,T,k,sigma,r,n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    R = np.exp(r * dt)
    p = (R - d) / (u - d)
    q = 1 - p
    ind = np.arange(n + 1)
    sech = s0 * u ** ind * d ** (n - ind)  # prix du ss jacent a l'echeance
    call = np.maximum(sech - k, 0)
    #la boucle ccle doit commencer de nieu n-1
    for i in range(n-1, -1, -1): #on descend d'un nieau
        ind = np.arange(i + 1)
        # dans les options eur pas besoin de tracabilite
        sech = s0 * u ** ind * d ** (n - ind)  # prix du ss jacent a l'echeance partiel
        call=[max(sech[j]-k,(p*call[j+1]+q*call[j])/R) for j in range(i+1) ]
    return call[0]

def binomialputam(s0,T,k,sigma,r,n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    R = np.exp(r * dt)
    p = (R - d) / (u - d)
    q = 1 - p
    ind = np.arange(n + 1)
    sech = s0 * u ** ind * d ** (n - ind)  # prix du ss jacent a l'echeance
    put = np.maximum(k-sech, 0)
    #la boucle ccle doit commencer de nieu n-1
    for i in range(n-1, -1, -1): #on descend d'un nieau
        ind = np.arange(i + 1)
        # dans les options am on est besoin de tracabilite , on fait tjr new boucles (pour chaque leel)
        sech = s0 * u ** ind * d ** (n - ind)  # prix du ss jacent a l'echeance partiel
        put=[max(k-sech[j],(p*put[j+1]+q*put[j])/R) for j in range(i+1) ]
    return put[0]

def fonctrep(s0,k,r,sigma,T):
   d1=(np.log(s0/k)+(r+1/2*sigma**2)*T)/(sigma*np.sqrt(T))
   d2=d1-sigma*np.sqrt(T)
   return d1,d2

def frCall(s0,k,r,T,sigma):
   d1,d2=fonctrep(s0,k,r,sigma,T)
   C=s0*norm.cdf(d1)-k*np.exp(-r*T)*norm.cdf(d2)
   return C

def frPut(s0,k,r,T,sigma):
   d1,d2=fonctrep(s0,k,r,sigma,T)
   P=k*np.exp(-r*T)*norm.cdf(-d2)-s0*norm.cdf(-d1)
   return P

import plotly.graph_objects as go
import numpy as np
import streamlit as st

def dessiner_arbre(s0, t, sigma, r, n):
    dt = t / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Liste pour stocker les positions et prix
    x_vals, y_vals, labels = [], [], []
    lines_x, lines_y = [], []

    # Étape 1 : construire les nœuds
    positions = {}
    for i in range(n + 1):
        for j in range(i + 1):
            s = s0 * (u ** j) * (d ** (i - j))
            x = i
            y = j - i / 2
            positions[(i, j)] = (x, y)

            x_vals.append(x)
            y_vals.append(y)
            labels.append(f"{s:.2f}")

    # Étape 2 : construire les arêtes
    for i in range(n):
        for j in range(i + 1):
            x0, y0 = positions[(i, j)]
            x_up, y_up = positions[(i + 1, j + 1)]
            x_down, y_down = positions[(i + 1, j)]

            # ligne montée
            lines_x += [x0, x_up, None]
            lines_y += [y0, y_up, None]

            # ligne descente
            lines_x += [x0, x_down, None]
            lines_y += [y0, y_down, None]

    # Création du graphique Plotly
    fig = go.Figure()

    # Arêtes
    fig.add_trace(go.Scatter(
        x=lines_x, y=lines_y,
        mode='lines',
        line=dict(color='black', width=1),
        hoverinfo='none',
        showlegend=False
    ))

    # Nœuds
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='markers+text',
        marker=dict(size=15, color='royalblue'),  # nœuds en bleu
        text=labels,
        textposition="top center",
        textfont=dict(color='black', size=10),   # texte en noir
        hoverinfo='text',
        showlegend=False
    ))

    fig.update_layout(
        title="Arbre binomial interactif",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',  # fond blanc
        paper_bgcolor='white',
        hovermode="closest",
        width=900, height=600
    )

    st.plotly_chart(fig, use_container_width=True)



import math


def trinomial_probabilities(sigma, r, q, dt):
    """
    Calcule les probabilités risk-neutral pour le modèle trinomial.

    sigma : volatilité du sous-jacent
    r     : taux d'intérêt sans risque
    q     : taux de dividende continu
    delta_t : pas de temps

    Retourne : (p_u, p_m, p_d)
    """
    # Facteurs
    u = math.exp(sigma * math.sqrt(2 * dt))
    d = 1 / u
    m = 1  # middle factor, généralement = 1

    # Probabilités risk-neutral
    pu = ((math.exp((r - q) * dt / 2) - math.exp(-sigma * math.sqrt(dt / 2))) /
          (math.exp(sigma * math.sqrt(dt / 2)) - math.exp(-sigma * math.sqrt(dt / 2)))) ** 2

    pd = ((math.exp(sigma * math.sqrt(dt / 2)) - math.exp((r - q) * dt / 2)) /
          (math.exp(sigma * math.sqrt(dt / 2)) - math.exp(-sigma * math.sqrt(dt / 2)))) ** 2

    pm = 1 - pu - pd

    return pu, pm, pd,u,d
def trinomialcalleur(sigma,r,q,s0,T,k,n):
    dt=T/n
    pu,pm,pd,u,d=trinomial_probabilities(sigma,r,q,dt)
    ind = np.arange(n+1)
    sech1=s0*d**(n-ind)
    ind2=np.arange(1,n+1)
    sech2=s0*u**ind2
    sech=np.concatenate((sech1, sech2))
    payoff=np.maximum(sech-k,0)
    call=payoff
    for i in range(2*n-2,-1,-2):
        call=[np.exp(-r*dt)*(pu*call[j+2]+pm*call[j+1]+pd*call[j]) for j in range(i+1)]
    return call[0]
def trinomialcallam(sigma,r,q,s0,T,k,n):
    dt=T/n
    pu,pm,pd,u,d=trinomial_probabilities(sigma,r,q,dt)
    ind = np.arange(n+1)
    sech1=s0*d**(n-ind)
    ind2=np.arange(1,n+1)
    sech2=s0*u**ind2
    sech=np.concatenate((sech1, sech2))
    payoff=np.maximum(sech-k,0)
    call=payoff
    for i in range(2*n-2,-1,-2):
        ind = np.arange(i/2+1)
        sech1 = s0 * d ** (n - ind)
        ind2 = np.arange(1, i /2+1)
        sech2 = s0 * u ** ind2
        sech = np.concatenate((sech1, sech2))
        payoff = np.maximum(sech - k, 0)
        call=[max(payoff[j],np.exp(-r*dt)*(pu*call[j+2]+pm*call[j+1]+pd*call[j])) for j in range(i+1)]
    return call[0]
def trinomialputeur(sigma,r,q,s0,T,k,n):
    dt=T/n
    pu,pm,pd,u,d=trinomial_probabilities(sigma,r,q,dt)
    ind = np.arange(n+1)
    sech1=s0*d**(n-ind)
    ind2=np.arange(1,n+1)
    sech2=s0*u**ind2
    sech=np.concatenate((sech1, sech2))
    payoff=np.maximum(k-sech,0)
    put=payoff
    for i in range(2*n-2,-1,-2):
        put=[np.exp(-r*dt)*(pu*put[j+2]+pm*put[j+1]+pd*put[j]) for j in range(i+1)]
    return put[0]
def trinomialputam(sigma,r,q,s0,T,k,n):
    dt=T/n
    pu,pm,pd,u,d=trinomial_probabilities(sigma,r,q,dt)
    ind = np.arange(n+1)
    sech1=s0*d**(n-ind)
    ind2=np.arange(1,n+1)
    sech2=s0*u**ind2
    sech=np.concatenate((sech1, sech2))
    payoff=np.maximum(k-sech,0)
    put=payoff
    for i in range(2*n-2,-1,-2):
        ind = np.arange(i/2+1)
        sech1 = s0 * d ** (n - ind)
        ind2 = np.arange(1, i /2+1)
        sech2 = s0 * u ** ind2
        sech = np.concatenate((sech1, sech2))
        payoff = np.maximum(k-sech, 0)
        put=[max(payoff[j],np.exp(-r*dt)*(pu*put[j+2]+pm*put[j+1]+pd*put[j])) for j in range(i+1)]
    return put[0]
def dessiner_arbre_trinomial(s0, T, sigma, r, q, n):
    dt = T / n
    pu, pm, pd, u, d = trinomial_probabilities(sigma, r, q, dt)

    positions = {}  # dictionnaire pour stocker les positions
    x_vals, y_vals, labels = [], [], []
    lines_x, lines_y = [], []

    # Construire les nœuds
    for i in range(n + 1):
        for j in range(2*i + 1):  # chaque niveau a 2*i + 1 nœuds
            # Calcul du prix selon le décalage par rapport au milieu
            s = s0 * (u ** max(0, j-i)) * (1 ** max(0, i-j)) * (d ** max(0, i-j))
            x = i
            y = j - i  # pour centrer verticalement
            positions[(i, j)] = (x, y)
            x_vals.append(x)
            y_vals.append(y)
            labels.append(f"{s:.2f}")

    # Construire les arêtes
    for i in range(n):
        for j in range(2*i + 1):
            x0, y0 = positions[(i, j)]
            # branches montée, milieu, descente
            for k in [j, j+1, j+2]:
                if (i+1, k) in positions:
                    x1, y1 = positions[(i+1, k)]
                    lines_x += [x0, x1, None]
                    lines_y += [y0, y1, None]

    # Création du graphique Plotly
    fig = go.Figure()

    # Arêtes
    fig.add_trace(go.Scatter(
        x=lines_x, y=lines_y,
        mode='lines',
        line=dict(color='black', width=1),
        hoverinfo='none',
        showlegend=False
    ))

    # Nœuds
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='markers+text',
        marker=dict(size=15, color='royalblue'),
        text=labels,
        textposition="top center",
        textfont=dict(color='black', size=10),
        hoverinfo='text',
        showlegend=False
    ))

    fig.update_layout(
        title="Arbre trinomial interactif",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode="closest",
        width=900, height=600
    )

    st.plotly_chart(fig, use_container_width=True)

