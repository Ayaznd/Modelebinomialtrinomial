import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from functions import *
import plotly.graph_objects as go  # à ajouter en haut du fichier
# --- Configuration de la page ---
st.set_page_config(page_title="Tarification des options", layout="wide")

# --- En-tête ---
st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'>Tarification des options</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; color: #3B82F6;'>Modèles binomial, trinomial et Black-Scholes</h3>",
    unsafe_allow_html=True
)
st.markdown("---")

# --- Définition des fonctions Black-Scholes ---
def fonctrep(s0, k, r, sigma, T):
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def frCall(s0, k, r, T, sigma):
    d1, d2 = fonctrep(s0, k, r, sigma, T)
    C = s0 * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
    return C

def frPut(s0, k, r, T, sigma):
    d1, d2 = fonctrep(s0, k, r, sigma, T)
    P = k * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
    return P
def main():
# --- Création des onglets ---
#tabs = st.tabs(["Modèle Binomial", "Modèle Trinomial", "Modèle Black-Scholes"])
 tabs = st.tabs(["Modèle Binomial", "Modèle Trinomial", "Modèle Black-Scholes", "Comparaison et Convergence","Analyse de sensibilité (Greeks)"])

# ==================================================
# ================= MODÈLE BINOMIAL ================
# ==================================================
 with tabs[0]:
    st.subheader("Modèle Binomial")
    st.markdown("### Paramètres")

    col1, col2 = st.columns(2)
    with col1:
        s0 = st.number_input("Prix initial de l'actif (S₀)", min_value=0.0, value=100.0)
        k = st.number_input("Prix d'exercice (K)", min_value=0.0, value=100.0)
        t = st.number_input("Échéance (T) en années", min_value=0.0, value=1.0)
    with col2:
        sigma = st.number_input("Volatilité (σ)", min_value=0.0, value=0.2)
        r = st.number_input("Taux d'intérêt (r)", min_value=0.0, value=0.05)
        n = st.number_input("Nombre de périodes (n)", min_value=1, value=50)

    Type = st.radio("Sélectionnez le type :", ["Call européen", "Put européen", "Call américain", "Put américain"], horizontal=True)

    colb1, colb2 = st.columns(2)
    with colb1:
        calcul = st.button("Calculer le prix (Binomial)")
    with colb2:
        arbre = st.button("Afficher l’arbre (Binomial)")

    if calcul:
        if Type == "Call européen":
            a = binomialcalleur(s0, t, k, sigma, r, n)
            st.success(f"Prix du call européen : {a:.2f}")
        elif Type == "Put européen":
            a = binomialputeur(s0, t, k, sigma, r, n)
            st.success(f"Prix du put européen : {a:.2f}")
        elif Type == "Call américain":
            a = binomialcallam(s0, t, k, sigma, r, n)
            st.success(f"Prix du call américain : {a:.2f}")
        elif Type == "Put américain":
            a = binomialputam(s0, t, k, sigma, r, n)
            st.success(f"Prix du put américain : {a:.2f}")

    if arbre:
        dessiner_arbre(s0, t, sigma, r, n)

# ==================================================
# ================= MODÈLE TRINOMIAL ===============
# ==================================================
 with tabs[1]:
    st.subheader("Modèle Trinomial")
    st.markdown("### Paramètres")

    col1, col2 = st.columns(2)
    with col1:
        s0 = st.number_input("Prix initial de l'actif (S₀)", min_value=0.0, value=100.0, key="s0_tri")
        k = st.number_input("Prix d'exercice (K)", min_value=0.0, value=100.0, key="k_tri")
        t = st.number_input("Échéance (T) en années", min_value=0.0, value=1.0, key="t_tri")
        sigma = st.number_input("Volatilité (σ)", min_value=0.0, value=0.2, key="sigma_tri")
    with col2:
        r = st.number_input("Taux d'intérêt (r)", min_value=0.0, value=0.05, key="r_tri")
        q = st.number_input("Taux de dividendes (q)", min_value=0.0, value=0.00, key="q_tri")
        n = st.number_input("Nombre de périodes (n)", min_value=1, value=50, key="n_tri")

    Type = st.radio("Sélectionnez le type :", ["Call européen", "Put européen","Call américain", "Put américain"], horizontal=True, key="type_tri")
    colb1, colb2 = st.columns(2)
    with colb1:
        calcul = st.button("Calculer le prix (Trinomial)")
    with colb2:
        arbre = st.button("Afficher l’arbre (Trinomial)")
    if calcul:
        if Type == "Call européen":
            a = trinomialcalleur(sigma, r, q, s0, t, k, n)
            st.success(f"Prix du call européen : {a:.2f}")
        elif Type == "Put européen":
            a = trinomialputeur(sigma, r, q, s0, t, k, n)
            st.success(f"Prix du put européen : {a:.2f}")
        elif Type == "Call américain":
            a = trinomialcallam(sigma,r,q,s0,t,k,n)
            st.success(f"Prix du call américain : {a:.2f}")
        elif Type == "Put américain":
            a = trinomialputam(sigma,r,q,s0,t,k,n)
            st.success(f"Prix du put américain : {a:.2f}")
    elif arbre:
        dessiner_arbre_trinomial(s0,t,sigma,r,q,n)

# ==================================================
# ============== MODÈLE BLACK-SCHOLES ==============
# ==================================================
 with tabs[2]:
    st.subheader("Modèle Black-Scholes")
    st.markdown("### Paramètres")

    col1, col2 = st.columns(2)
    with col1:
        s0 = st.number_input("Prix initial de l'actif (S₀)", min_value=0.0, value=100.0, key="s0_bs")
        k = st.number_input("Prix d'exercice (K)", min_value=0.0, value=100.0, key="k_bs")
        sigma = st.number_input("Volatilité (σ)", min_value=0.0, value=0.2, key="sigma_bs")
    with col2:
        T = st.number_input("Échéance (T) en années", min_value=0.0, value=1.0, key="T_bs")
        r = st.number_input("Taux d'intérêt (r)", min_value=0.0, value=0.05, key="r_bs")
    colbs1, colbs2 = st.columns(2)
    with colbs1:
        if st.button("Calculer Call européen"):
            call_price = frCall(s0, k, r, T, sigma)
            st.success(f"Prix du call européen (Black-Scholes) : {call_price:.2f}")

    with colbs2:
        if st.button("Calculer Put européen"):
            put_price = frPut(s0, k, r, T, sigma)
            st.success(f"Prix du put européen (Black-Scholes) : {put_price:.2f}")

# ==================================================
# =========== COMPARAISON ET CONVERGENCE ===========
# ==================================================
 with tabs[3]:
    st.subheader("Comparaison et Convergence des modèles")
    st.markdown("Visualisation de la convergence des modèles binomial et trinomial vers Black-Scholes quand le nombre de périodes augmente.")

    col1, col2 = st.columns(2)
    with col1:
        s0 = st.number_input("Prix initial de l'actif (S₀)", min_value=0.0, value=100.0, key="s0_comp")
        k = st.number_input("Prix d'exercice (K)", min_value=0.0, value=100.0, key="k_comp")
        T = st.number_input("Échéance (T) en années", min_value=0.0, value=1.0, key="T_comp")
    with col2:
        sigma = st.number_input("Volatilité (σ)", min_value=0.0, value=0.2, key="sigma_comp")
        r = st.number_input("Taux d'intérêt (r)", min_value=0.0, value=0.05, key="r_comp")

    Type = st.radio("Type d'option :", ["Call européen", "Put européen"], horizontal=True, key="type_comp")



    if st.button("Afficher la convergence"):
        n_values = np.arange(1, 100, 5)
        prix_binomial, prix_trinomial = [], []

        for n in n_values:
            if Type == "Call européen":
                prix_binomial.append(binomialcalleur(s0, T, k, sigma, r, n))
                prix_trinomial.append(trinomialcalleur(sigma, r, 0, s0, T, k, n))
                bs_price = frCall(s0, k, r, T, sigma)
            else:
                prix_binomial.append(binomialputeur(s0, T, k, sigma, r, n))
                prix_trinomial.append(trinomialputeur(sigma, r, 0, s0, T, k, n))
                bs_price = frPut(s0, k, r, T, sigma)

        # Création du graphique interactif
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=n_values, y=prix_binomial,
            mode='lines+markers',
            name='Binomial',
            line=dict(dash='dash', color='#1E3A8A'),
            marker=dict(size=5)
        ))

        fig.add_trace(go.Scatter(
            x=n_values, y=prix_trinomial,
            mode='lines+markers',
            name='Trinomial',
            line=dict(dash='dot', color='#3B82F6'),
            marker=dict(size=5)
        ))

        fig.add_hline(
            y=bs_price, line=dict(color='red', width=2),
            annotation_text='Black-Scholes',
            annotation_position='bottom right'
        )

        fig.update_layout(
            title="Convergence des modèles vers Black-Scholes",
            xaxis_title="Nombre de périodes (n)",
            yaxis_title="Prix de l'option",
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)")
        )

        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# =========== ANALYSE DE SENSIBILITÉ ===============
# ==================================================
 with tabs[4]:
    st.subheader("Analyse de sensibilité (Greeks)")
    st.markdown("""
    Cette section permet de visualiser comment le **prix d'une option** et ses **Greeks** réagissent aux variations du prix du sous-jacent.
    Les Greeks mesurent la sensibilité du prix d'une option à différents paramètres :
    - **Δ (Delta)** : variation du prix de l’option selon le sous-jacent  
    - **Γ (Gamma)** : variation du Delta  
    - **Vega** : sensibilité à la volatilité  
    - **Θ (Theta)** : sensibilité au temps  
    - **ρ (Rho)** : sensibilité au taux d’intérêt  
    """)

    col1, col2 = st.columns(2)
    with col1:
        s0 = st.number_input("Prix initial (S₀)", min_value=0.0, value=100.0, key="s0_greeks")
        k = st.number_input("Prix d'exercice (K)", min_value=0.0, value=100.0, key="k_greeks")
        T = st.number_input("Échéance (T)", min_value=0.01, value=1.0, key="T_greeks")
    with col2:
        sigma = st.number_input("Volatilité (σ)", min_value=0.0, value=0.2, key="sigma_greeks")
        r = st.number_input("Taux d'intérêt (r)", min_value=0.0, value=0.05, key="r_greeks")
        Type = st.radio("Type d'option :", ["Call", "Put"], horizontal=True, key="type_greeks")

    greek_to_plot = st.selectbox(
        "Choisir le Greek à visualiser :",
        ["Prix de l'option", "Delta (Δ)", "Gamma (Γ)", "Vega", "Theta (Θ)", "Rho (ρ)"],
        index=0
    )

    if st.button("Afficher les graphiques"):
        s_values = np.linspace(0.5 * s0, 1.5 * s0, 100)
        prix, delta, gamma, vega, theta, rho = [], [], [], [], [], []

        for s in s_values:
            d1, d2 = fonctrep(s, k, r, sigma, T)

            if Type == "Call":
                prix.append(frCall(s, k, r, T, sigma))
                delta.append(norm.cdf(d1))
                gamma.append(norm.pdf(d1) / (s * sigma * np.sqrt(T)))
                vega.append(s * norm.pdf(d1) * np.sqrt(T))
                theta.append(-(s * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * k * np.exp(-r * T) * norm.cdf(d2))
                rho.append(k * T * np.exp(-r * T) * norm.cdf(d2))
            else:
                prix.append(frPut(s, k, r, T, sigma))
                delta.append(-norm.cdf(-d1))
                gamma.append(norm.pdf(d1) / (s * sigma * np.sqrt(T)))
                vega.append(s * norm.pdf(d1) * np.sqrt(T))
                theta.append(-(s * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * k * np.exp(-r * T) * norm.cdf(-d2))
                rho.append(-k * T * np.exp(-r * T) * norm.cdf(-d2))

        # Sélection dynamique du Greek à afficher
        fig = go.Figure()

        if greek_to_plot == "Prix de l'option":
            y = prix
            title = f"Prix de l'option {Type} en fonction de S"
        elif greek_to_plot == "Delta (Δ)":
            y = delta
            title = f"Delta (Δ) en fonction de S"
        elif greek_to_plot == "Gamma (Γ)":
            y = gamma
            title = f"Gamma (Γ) en fonction de S"
        elif greek_to_plot == "Vega":
            y = vega
            title = f"Vega en fonction de S"
        elif greek_to_plot == "Theta (Θ)":
            y = theta
            title = f"Theta (Θ) en fonction de S"
        else:
            y = rho
            title = f"Rho (ρ) en fonction de S"

        fig.add_trace(go.Scatter(
            x=s_values,
            y=y,
            mode='lines',
            name=greek_to_plot,
            line=dict(color="#3B82F6", width=3)
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Prix du sous-jacent (S)",
            yaxis_title=greek_to_plot,
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

if __name__=="__main__":
    main()
