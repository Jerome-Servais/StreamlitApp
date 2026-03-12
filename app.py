import streamlit as st
import pandas as pd
from components import exploration,visualisations,model

st.set_page_config(page_title="House Prices Explorer", layout="wide")
st.title("House Prices Explorer")
st.markdown("Explorez, visualisez et prédisez les prix de vente de logements à partir du dataset Kaggle *House Prices*.")



if True:

    # ── 1. CHARGEMENT & CACHE ──────────────────────────────────────────────────
    @st.cache_data
    def load_and_prepare(file):
        df = pd.read_csv(file)
        current_year = 2026
        df["AgeLogement"]    = current_year - df["YearBuilt"]
        df["SurfaceTotale"]  = df["GrLivArea"] + df["TotalBsmtSF"]
        df["NbSallesDeBain"] = df["FullBath"] + 0.5 * df["HalfBath"]
        return df

    df = load_and_prepare("data/train.csv")

    # ── 2. SIDEBAR – FILTRES ───────────────────────────────────────────────────
    with st.sidebar:
        st.header(" Filtres")

        # Filtre 1 – Plage de prix
        price_min, price_max = int(df["SalePrice"].min()), int(df["SalePrice"].max())
        price_range = st.slider(
            "Prix de vente ($)",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
            step=1_000,
            help="Sélectionnez la plage de prix de vente souhaitée (en dollars US).",
        )

        # Filtre 2 – Quartier
        neighborhoods = ["Tous"] + sorted(df["Neighborhood"].unique().tolist())
        selected_neighborhood = st.selectbox(
            "Quartier (Neighborhood)",
            options=neighborhoods,
            help="Filtrez les logements selon leur quartier d'appartenance.",
        )

        # Filtre 3 – Âge maximum
        age_max = st.slider(
            "Âge maximum du logement (années)",
            min_value=0,
            max_value=int(df["AgeLogement"].max()),
            value=int(df["AgeLogement"].max()),
            help="Affichez uniquement les logements construits il y a moins de N années.",
        )

        # Filtre 4 – Qualité générale minimale
        qual_min = st.slider(
            "Qualité générale minimale (1 – 10)",
            min_value=1,
            max_value=10,
            value=1,
            help="Filtrez par note de qualité globale attribuée au logement (1 = très mauvais, 10 = excellent).",
        )

        st.divider()
        st.caption("Les filtres s'appliquent à tous les onglets.")

    # ── Application des filtres ────────────────────────────────────────────────
    filtered = df[
        (df["SalePrice"]   >= price_range[0]) &
        (df["SalePrice"]   <= price_range[1]) &
        (df["AgeLogement"] <= age_max)        &
        (df["OverallQual"] >= qual_min)
    ]
    if selected_neighborhood != "Tous":
        filtered = filtered[filtered["Neighborhood"] == selected_neighborhood]

    # ── 3. ONGLETS ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Exploration", " Visualisations", "Modèle"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – EXPLORATION
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        exploration.exploration(filtered)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – VISUALISATIONS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        visualisations.visu(filtered)
        

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – MODÈLE
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        model.predict_tab(df)

else:
    st.info(" Veuillez charger le fichier **train.csv** pour démarrer l'exploration.")
    st.markdown(
        """
        **Dataset attendu :** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)  
        Colonnes minimales requises : `SalePrice`, `YearBuilt`, `GrLivArea`, `TotalBsmtSF`, `FullBath`, `HalfBath`,
        `Neighborhood`, `OverallQual`, `LotArea`, `BedroomAbvGr`.
        """
    )
