import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def exploration(filtered):
    st.header("Exploration des données")

    # Bandeau de métriques
    n = len(filtered)
    st.info(f"**{n} logement{'s' if n > 1 else ''}** correspondent aux filtres sélectionnés.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label=" Prix moyen",
        value=f"${filtered['SalePrice'].mean():,.0f}",
        help="Prix de vente moyen des logements filtrés.",
    )
    col2.metric(
        label=" Surface moyenne",
        value=f"{filtered['SurfaceTotale'].mean():,.0f} ft²",
        help="Surface habitable totale (rez-de-chaussée + sous-sol) moyenne.",
    )
    col3.metric(
        label=" Âge moyen",
        value=f"{filtered['AgeLogement'].mean():.0f} ans",
        help="Âge moyen des logements filtrés (calculé depuis 2026).",
    )
    col4.metric(
        label=" Qualité moyenne",
        value=f"{filtered['OverallQual'].mean():.1f} / 10",
        help="Note de qualité générale moyenne des logements filtrés.",
    )

    st.subheader("Aperçu du tableau de données")
    display_cols = [
        "Neighborhood", "SalePrice", "SurfaceTotale",
        "AgeLogement", "NbSallesDeBain", "OverallQual", "LotArea",
    ]
    st.dataframe(
        filtered[display_cols]
        .rename(columns={
            "Neighborhood":   "Quartier",
            "SalePrice":      "Prix ($)",
            "SurfaceTotale":  "Surface totale (ft²)",
            "AgeLogement":    "Âge (ans)",
            "NbSallesDeBain": "Salles de bain",
            "OverallQual":    "Qualité (1-10)",
            "LotArea":        "Terrain (ft²)",
        })
        .head(100),
        use_container_width=True,
    )

    # Téléchargement CSV
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger les données filtrées (CSV)",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv",
        help="Exporte les données actuellement affichées au format CSV.",
    )