import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import plotly.express as px



# ══════════════════════════════════════════════════════════════════════════
# TAB 3 – MODÈLE
# ══════════════════════════════════════════════════════════════════════════
def predict_tab(data):
    st.header("Modèles de prédiction & Comparaison")

    FEATURES = [
        "SurfaceTotale", "AgeLogement", "NbSallesDeBain",
        "OverallQual", "LotArea", "BedroomAbvGr",
    ]
    FEATURE_LABELS = {
        "SurfaceTotale":  "Surface totale (ft²)",
        "AgeLogement":    "Âge du logement (ans)",
        "NbSallesDeBain": "Nombre de salles de bain",
        "OverallQual":    "Qualité générale (1-10)",
        "LotArea":        "Surface du terrain (ft²)",
        "BedroomAbvGr":   "Nombre de chambres",
    }

    MODEL_OPTIONS = {
        "Gradient Boosting":  GradientBoostingRegressor(random_state=42),
        "Régression Linéaire": LinearRegression(),
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
    }

    MODEL_COLORS = {
        "Gradient Boosting":  "#2563EB",
        "Régression Linéaire": "#16A34A",
        "Random Forest":      "#D97706",
    }

    # ── Entraînement de tous les modèles ──────────────────────────────────────
    @st.cache_resource
    def train_all(data):
        X = data[FEATURES].fillna(data[FEATURES].median())
        y = data["SalePrice"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        results = {}
        for name, mdl in MODEL_OPTIONS.items():
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_val)
            results[name] = {
                "model":  mdl,
                "y_val":  y_val,
                "y_pred": y_pred,
                "r2":     r2_score(y_val, y_pred),
                "mae":    mean_absolute_error(y_val, y_pred),
                "rmse":   np.sqrt(mean_squared_error(y_val, y_pred)),
            }
        return results, X_val, y_val

    with st.spinner("Entraînement des modèles en cours…"):
        all_results, X_val, y_val = train_all(data)

    # ── Sélecteur de modèle actif ─────────────────────────────────────────────
    st.subheader("Choisir un modèle à analyser")
    selected_model = st.radio(
        label="Modèle actif",
        options=list(MODEL_OPTIONS.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    res = all_results[selected_model]

    # ── Métriques du modèle sélectionné ───────────────────────────────────────
    st.subheader(f"Performance — {selected_model}")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="R² Score",
        value=f"{res['r2']:.3f}",
        help="Coefficient de détermination : 1.0 = prédiction parfaite.",
    )
    col2.metric(
        label="MAE – Erreur absolue moyenne",
        value=f"${res['mae']:,.0f}",
        help="Écart moyen (en valeur absolue) entre prix réel et prix prédit.",
    )
    col3.metric(
        label="RMSE – Racine de l'erreur quadratique",
        value=f"${res['rmse']:,.0f}",
        help="Pénalise davantage les grandes erreurs de prédiction.",
    )

    # ── Graphe réel vs prédit ─────────────────────────────────────────────────
    fig_pred = px.scatter(
        x=res["y_val"],
        y=res["y_pred"],
        labels={"x": "Prix réel ($)", "y": "Prix prédit ($)"},
        title=f"Prix réels vs Prix prédits — {selected_model}",
        opacity=0.65,
        color_discrete_sequence=[MODEL_COLORS[selected_model]],
    )
    fig_pred.add_shape(
        type="line",
        x0=y_val.min(), y0=y_val.min(),
        x1=y_val.max(), y1=y_val.max(),
        line=dict(color="#DC2626", dash="dash", width=2),
    )
    fig_pred.update_layout(xaxis_tickformat="$,.0f", yaxis_tickformat="$,.0f")
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Importance des features (si disponible) ───────────────────────────────
    mdl_obj = res["model"]
    if hasattr(mdl_obj, "feature_importances_"):
        importances = mdl_obj.feature_importances_
        fig_imp = px.bar(
            x=importances,
            y=[FEATURE_LABELS[f] for f in FEATURES],
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            title=f"Importance des variables — {selected_model}",
            color=importances,
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)
    elif hasattr(mdl_obj, "coef_"):
        coefs = np.abs(mdl_obj.coef_)
        fig_coef = px.bar(
            x=coefs,
            y=[FEATURE_LABELS[f] for f in FEATURES],
            orientation="h",
            labels={"x": "Coefficient (valeur absolue)", "y": "Feature"},
            title=f"Coefficients — {selected_model}",
            color=coefs,
            color_continuous_scale="Greens",
        )
        fig_coef.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_coef, use_container_width=True)

    # ── Tableau comparatif des 3 modèles ──────────────────────────────────────
    st.divider()
    st.subheader("Comparaison des 3 modèles")

    comparison_df = pd.DataFrame([
        {
            "Modèle": name,
            "R²": f"{r['r2']:.3f}",
            "MAE ($)": f"{r['mae']:,.0f}",
            "RMSE ($)": f"{r['rmse']:,.0f}",
        }
        for name, r in all_results.items()
    ])

    # Mise en évidence du meilleur R²
    best_r2 = max(all_results, key=lambda n: all_results[n]["r2"])

    def highlight_best(row):
        if row["Modèle"] == best_r2:
            return ["background-color: #DBEAFE; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        comparison_df.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Meilleur modèle (R² le plus élevé) : **{best_r2}** — surligné en bleu.")

    # Graphe comparatif R² / MAE / RMSE
    metric_choice = st.selectbox(
        "Métrique à comparer",
        options=["R²", "MAE ($)", "RMSE ($)"],
        index=0,
    )
    metric_key_map = {"R²": "r2", "MAE ($)": "mae", "RMSE ($)": "rmse"}
    mk = metric_key_map[metric_choice]
    bar_vals = [all_results[n][mk] for n in all_results]
    bar_colors = [MODEL_COLORS[n] for n in all_results]

    fig_comp = px.bar(
        x=list(all_results.keys()),
        y=bar_vals,
        labels={"x": "Modèle", "y": metric_choice},
        title=f"Comparaison — {metric_choice}",
        color=list(all_results.keys()),
        color_discrete_map=MODEL_COLORS,
    )
    fig_comp.update_layout(showlegend=False)
    if metric_choice != "R²":
        fig_comp.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Formulaire de prédiction ───────────────────────────────────────────────
    st.divider()
    st.subheader("Faire une prédiction")
    st.caption(
        "Renseignez les caractéristiques du logement. "
        "Les estimations des 3 modèles seront affichées simultanément."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        surface = st.number_input(
            "Surface totale (ft²)",
            min_value=500, max_value=10_000, value=2_000, step=50,
            help="Surface habitable totale incluant le rez-de-chaussée et le sous-sol.",
        )
        age = st.number_input(
            "Âge du logement (années)",
            min_value=0, max_value=150, value=20,
            help="Nombre d'années depuis la construction du logement.",
        )
    with col2:
        bains = st.number_input(
            "Nombre de salles de bain",
            min_value=0.5, max_value=5.0, value=2.0, step=0.5,
            help="Salles de bain complètes = 1 ; demi-salles = 0.5.",
        )
        qualite = st.slider(
            "Qualité générale (1 – 10)",
            min_value=1, max_value=10, value=5,
            help="1 = très mauvais, 10 = excellent.",
        )
    with col3:
        lot = st.number_input(
            "Surface du terrain (ft²)",
            min_value=1_000, max_value=50_000, value=8_000, step=500,
        )
        chambres = st.number_input(
            "Nombre de chambres (hors sous-sol)",
            min_value=0, max_value=10, value=3,
        )

    if st.button("Estimer le prix", type="primary"):
        input_df = pd.DataFrame(
            [[surface, age, bains, qualite, lot, chambres]],
            columns=FEATURES,
        )

        st.markdown("#### Estimations par modèle")
        pred_cols = st.columns(3)
        for i, (name, r) in enumerate(all_results.items()):
            pred = r["model"].predict(input_df)[0]
            pred_cols[i].metric(
                label=name,
                value=f"${pred:,.0f}",
                help=f"R² = {r['r2']:.3f}",
            )

        # Graphe des 3 estimations
        pred_vals = [all_results[n]["model"].predict(input_df)[0] for n in all_results]
        fig_preds = px.bar(
            x=list(all_results.keys()),
            y=pred_vals,
            labels={"x": "Modèle", "y": "Prix estimé ($)"},
            title="Comparaison des estimations",
            color=list(all_results.keys()),
            color_discrete_map=MODEL_COLORS,
            text=[f"${v:,.0f}" for v in pred_vals],
        )
        fig_preds.update_traces(textposition="outside")
        fig_preds.update_layout(showlegend=False, yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig_preds, use_container_width=True)
