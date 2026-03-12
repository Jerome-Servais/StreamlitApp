import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


def visu(filtered):
    st.header("Visualisations")

    if filtered.empty:
        st.warning("Aucune donnée à afficher. Veuillez ajuster les filtres dans la barre latérale.")
    else:
        # Ligne 1 : histogramme + scatter
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribution des prix de vente")
            fig_hist = px.histogram(
                filtered,
                x="SalePrice",
                nbins=50,
                labels={"SalePrice": "Prix de vente ($)", "count": "Nombre de logements"},
                color_discrete_sequence=["#2563EB"],  # bleu accessible
                title="Répartition des prix de vente",
            )
            fig_hist.update_layout(bargap=0.05, xaxis_tickformat="$,.0f")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.subheader("Surface totale vs Prix de vente")
            fig_scatter = px.scatter(
                filtered,
                x="SurfaceTotale",
                y="SalePrice",
                color="OverallQual",
                labels={
                    "SurfaceTotale": "Surface totale (ft²)",
                    "SalePrice":     "Prix de vente ($)",
                    "OverallQual":   "Qualité (1-10)",
                },
                color_continuous_scale="Viridis",
                title="Surface vs Prix, coloré par qualité",
                opacity=0.75,
            )
            fig_scatter.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Ligne 2 : box plot + corrélation
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Prix de vente par quartier")
            top_n = filtered["Neighborhood"].value_counts().head(15).index
            fig_box = px.box(
                filtered[filtered["Neighborhood"].isin(top_n)],
                x="Neighborhood",
                y="SalePrice",
                labels={
                    "SalePrice":     "Prix de vente ($)",
                    "Neighborhood":  "Quartier",
                },
                title="Distribution des prix par quartier (top 15)",
                color_discrete_sequence=["#7C3AED"],  # violet accessible
            )
            fig_box.update_xaxes(tickangle=45)
            fig_box.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(fig_box, use_container_width=True)

        with col4:
            st.subheader("Matrice de corrélation")
            num_cols = [
                "SalePrice", "SurfaceTotale", "AgeLogement",
                "NbSallesDeBain", "OverallQual", "LotArea",
            ]
            corr_labels = {
                "SalePrice":      "Prix",
                "SurfaceTotale":  "Surface",
                "AgeLogement":    "Âge",
                "NbSallesDeBain": "Salles de bain",
                "OverallQual":    "Qualité",
                "LotArea":        "Terrain",
            }
            corr = filtered[num_cols].corr()
            corr.index   = [corr_labels[c] for c in corr.index]
            corr.columns = [corr_labels[c] for c in corr.columns]

            fig_corr, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title("Corrélations entre variables numériques", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_corr)