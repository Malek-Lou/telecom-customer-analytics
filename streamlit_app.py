import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Telecom Customer Analytics", layout="wide")

# ==============================================================================
# CHARGEMENT DES DONNEES
# ==============================================================================

st.sidebar.header("Donnees")
uploaded = st.sidebar.file_uploader("Fichier clients_export.csv", type="csv")
uploaded_models = st.sidebar.file_uploader(
    "Fichier model_comparison.csv (optionnel)", type="csv", key="models_upl"
)
uploaded_importance = st.sidebar.file_uploader(
    "Fichier feature_importance.csv (optionnel)", type="csv", key="importance_upl"
)

DEFAULT_PATH = "demo_data/clients_export.csv"
DEFAULT_MODELS_PATH = "demo_data/model_comparison.csv"
DEFAULT_IMPORTANCE_PATH = "demo_data/feature_importance.csv"


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


def try_load_optional(uploaded_file, default_path):
    """Charge un CSV optionnel (upload en priorite, sinon fichier par defaut)."""
    if uploaded_file is not None:
        return load_data(uploaded_file)
    try:
        return load_data(default_path)
    except FileNotFoundError:
        return None


if uploaded is not None:
    df = load_data(uploaded)
    demo_mode = False
else:
    try:
        df = load_data(DEFAULT_PATH)
        demo_mode = True
        st.sidebar.caption("Jeu de demonstration synthetique charge par defaut.")
    except FileNotFoundError:
        st.warning(
            "Aucun fichier charge. Depose 'clients_export.csv' via le panneau de gauche "
            "ou genere les fichiers publics avec `python generate_demo_data.py`."
        )
        st.stop()

model_comparison_df = try_load_optional(uploaded_models, DEFAULT_MODELS_PATH)
feature_importance_df = try_load_optional(uploaded_importance, DEFAULT_IMPORTANCE_PATH)

st.sidebar.header("Filtres")

if "Cluster_KMeans" in df.columns:
    clusters = sorted(df["Cluster_KMeans"].dropna().unique().tolist())
    selected_clusters = st.sidebar.multiselect("Segment (cluster K-Means)", clusters, default=clusters)
    df = df[df["Cluster_KMeans"].isin(selected_clusters)]

if "Segment_Avance" in df.columns:
    segments_av = sorted(df["Segment_Avance"].dropna().unique().tolist())
    selected_segments_av = st.sidebar.multiselect("Segment comportemental (RFM)", segments_av, default=segments_av)
    df = df[df["Segment_Avance"].isin(selected_segments_av)]

if "Main_Offer" in df.columns:
    offers = sorted(df["Main_Offer"].dropna().unique().tolist())
    selected_offers = st.sidebar.multiselect("Offre principale", offers, default=offers)
    df = df[df["Main_Offer"].isin(selected_offers)]

if "Churn_Proba" in df.columns:
    risk_threshold = st.sidebar.slider("Seuil de risque de churn", 0.0, 1.0, 0.5, 0.05)
else:
    risk_threshold = None

st.title("Dashboard — Telecom Customer Analytics")

if demo_mode:
    st.info(
        "Synthetic demonstration data — no confidential company or customer data is displayed."
    )
else:
    st.warning(
        "Local uploaded data are being displayed. Do not publish screenshots or exports "
        "unless you have verified that the data are safe to share."
    )

st.caption(f"{len(df):,} clients apres filtres")

SEGMENT_COLORS = {
    "Champions": "#2ca02c",
    "Champion": "#2ca02c",
    "Clients fidèles": "#1f77b4",
    "Client fidèle": "#1f77b4",
    "Client moyen": "#8c8c8c",
    "Clients prometteurs": "#17becf",
    "Clients à développer": "#ff9f1c",
    "Ne doit pas les perdre": "#9467bd",
    "Clients à risque": "#d62728",
    "Client à risque": "#d62728",
    "Clients hibernants": "#7f7f7f",
    "Nouveaux clients": "#e377c2",
}


def color_map_for(values):
    """Renvoie une palette coherente pour une liste de segments, avec fallback."""
    palette = px.colors.qualitative.Set2
    cmap = {}
    for i, v in enumerate(sorted(set(values))):
        cmap[v] = SEGMENT_COLORS.get(v, palette[i % len(palette)])
    return cmap


def display_customer_table(frame):
    """Prepare a customer table for display without changing the source schema."""
    shown = frame.copy()
    if "msisdn_crypte" in shown.columns:
        shown = shown.rename(columns={"msisdn_crypte": "Customer_ID"})
    return shown


# ==============================================================================
# NAVIGATION PAR ONGLETS - on suit le fil du notebook, comme une histoire
# ==============================================================================

tab_story, tab_overview, tab_rfm, tab_behavior, tab_cluster, tab_churn, tab_model, tab_offers = st.tabs([
    "Histoire de l'analyse",
    "Vue d'ensemble",
    "RFM",
    "Segments comportementaux",
    "Clustering K-Means",
    "Risque de churn",
    "Modele predictif",
    "Offres & revenus",
])

# ------------------------------------------------------------------------------
# ONGLET 1 : L'HISTOIRE DE L'ANALYSE (resume narratif du notebook)
# ------------------------------------------------------------------------------
with tab_story:
    st.header("De la donnee brute au score de churn : le fil de l'analyse")

    st.markdown(
        """
Ce dashboard met en scene le travail fait dans le notebook `analyse.ipynb` sur les
clients prepayes d'un operateur telecom. Voici, dans l'ordre, ce qui a ete fait.
"""
    )

    st.subheader("1. Preparation des donnees")
    st.markdown(
        """
Le point de depart est un export transactionnel brut (une ligne par achat/option,
avec un identifiant client crypte `msisdn_crypte`, des dates d'activation/modification/
desactivation/achat, une offre, un montant...).

- **Valeurs manquantes** : la colonne `life_time` avait des trous, comble a partir de
  la date d'activation quand elle etait disponible.
- **Doublons** : detectes et supprimes.
- **Types** : dates converties en `datetime`, identifiants et offres en categories,
  `montant` et `life_time` forces en numerique.
"""
    )

    st.subheader("2. Exploration (EDA)")
    st.markdown(
        """
Avant de segmenter qui que ce soit, on regarde la donnee : distribution du montant et
de l'anciennete, top offres et top options utilisees, correlation entre variables
numeriques, et evolution mensuelle des activations / modifications / desactivations
dans le temps.
"""
    )

    st.subheader("3. RFM : Recence, Frequence, Montant")
    st.markdown(
        """
Chaque client est resume par 3 chiffres calcules a partir de son historique d'achat :

- **Recency** : nombre de jours depuis le dernier achat.
- **Frequency** : nombre d'achats.
- **Monetary** : montant total depense.

Ces 3 valeurs sont ensuite converties en scores de 1 a 5 (quintiles) puis combinees
en un score RFM global, utilise pour classer les clients en segments simples
(*Champion*, *Client fidele*, *Client moyen*, *Client a risque*).

Un score **pondere** (poids 0.5 pour le montant, 0.3 pour la frequence, 0.2 pour la
recence) est egalement compare au score simple R+F+M afin d'etudier l'impact de la
ponderation sur le classement des clients.
"""
    )

    st.subheader("4. Segmentation comportementale avancee")
    st.markdown(
        """
Au-dela du score global, un segment plus fin est construit a partir des scores R et F
seuls (Champions, Clients fideles, Clients prometteurs, Clients a developper, Ne doit
pas les perdre, Clients a risque, Clients hibernants), avec une categorie dediee aux
**Nouveaux clients** selon l'anciennete. Le profil de chaque segment permet ensuite de
comparer le nombre de clients, la recence, la frequence, le montant et la contribution
au revenu. Une courbe de Lorenz est utilisee pour analyser la concentration de la valeur.
"""
    )

    st.subheader("5. Feature engineering avant clustering")
    st.markdown(
        """
De nouvelles variables sont derivees : `Life_Time` (anciennete moyenne), `Montant_
Moyen_Par_Achat`, ainsi que `Frequency_Par_Mois` et `Monetary_Par_Mois` (normalisees
par l'anciennete, pour comparer equitablement un client recent et un client ancien).
"""
    )

    st.subheader("6. Clustering K-Means")
    st.markdown(
        """
Les variables numeriques sont standardisees (`StandardScaler`). Une PCA est utilisee
pour projeter les profils clients en deux dimensions et faciliter la visualisation.
Plusieurs valeurs de k sont evaluees a l'aide de l'inertie, du score de silhouette et
de l'indice de Davies-Bouldin. La stabilite du clustering est egalement etudiee avec
plusieurs initialisations et l'Adjusted Rand Index (ARI). DBSCAN est explore comme
methode complementaire de clustering.
"""
    )

    st.subheader("7. Definition et modelisation du churn")
    st.markdown(
        """
Le churn est defini a partir d'un seuil metier configure dans l'analyse. Trois modeles
de classification sont compares a partir des variables RFM et derivees :
**Random Forest**, **XGBoost** et **LightGBM**. Les hyperparametres sont recherches avec
`GridSearchCV` et validation croisee. Les modeles sont compares avec plusieurs
metriques, notamment l'accuracy, le F1-score et l'AUC-ROC. Des scores hors echantillon
sont ensuite exportes dans `Churn_Proba`, et l'importance des variables est analysee
pour faciliter l'interpretation du modele.
"""
    )

    st.subheader("8. Export pour ce dashboard")
    st.markdown(
        """
Le notebook termine par l'export d'un fichier `clients_export.csv` (un client par
ligne, avec ses scores RFM, ses segments, son cluster, sa probabilite de churn et ses
coordonnees PCA), accompagne de `model_comparison.csv` et `feature_importance.csv`
pour que ce dashboard puisse rejouer l'integralite de l'histoire ci-dessus,
interactivement.
"""
    )

    st.info(
        "Utilise les onglets ci-dessus pour parcourir chaque etape avec les graphiques "
        "interactifs correspondants."
    )

# ------------------------------------------------------------------------------
# ONGLET 2 : VUE D'ENSEMBLE
# ------------------------------------------------------------------------------
with tab_overview:
    st.header("Vue d'ensemble")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Clients", f"{len(df):,}")

    if "Churned" in df.columns:
        churn_rate = df["Churned"].mean() * 100
        col2.metric("Taux de churn observe", f"{churn_rate:.1f}%")
    else:
        col2.metric("Taux de churn observe", "N/A")

    if "Monetary" in df.columns:
        col3.metric("Monetary moyen", f"{df['Monetary'].mean():,.0f}")
    else:
        col3.metric("Monetary moyen", "N/A")

    if "Cluster_KMeans" in df.columns:
        col4.metric("Nb segments (clusters)", f"{df['Cluster_KMeans'].nunique()}")
    else:
        col4.metric("Nb segments (clusters)", "N/A")

    if "Tenure_Months" in df.columns:
        col5.metric("Anciennete moyenne (mois)", f"{df['Tenure_Months'].mean():.1f}")
    else:
        col5.metric("Anciennete moyenne (mois)", "N/A")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if "Nb_Offers_Distinctes" in df.columns:
            fig_nb_offers = px.histogram(
                df, x="Nb_Offers_Distinctes", nbins=20,
                title="Distribution du nombre d'offres distinctes par client",
                color_discrete_sequence=["darkorange"],
            )
            st.plotly_chart(fig_nb_offers, use_container_width=True)

    with c2:
        if "Life_Time" in df.columns:
            fig_life = px.histogram(
                df, x="Life_Time", nbins=30,
                title="Distribution de l'anciennete moyenne (Life_Time)",
                color_discrete_sequence=["steelblue"],
            )
            st.plotly_chart(fig_life, use_container_width=True)

    num_cols_corr = [c for c in [
        "Recency", "Frequency", "Monetary", "Tenure_Months", "Nb_Offers_Distinctes",
        "Life_Time", "Montant_Moyen_Par_Achat", "Frequency_Par_Mois", "Monetary_Par_Mois",
        "Churn_Proba",
    ] if c in df.columns]

    if len(num_cols_corr) >= 2:
        st.subheader("Correlation entre variables numeriques")
        corr = df[num_cols_corr].corr().round(2)
        fig_corr = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Matrice de correlation",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------------------------------------------------------------
# ONGLET 3 : ANALYSE RFM
# ------------------------------------------------------------------------------
with tab_rfm:
    st.header("Analyse RFM (Recence - Frequence - Montant)")

    rfm_cols_present = [c for c in ["Recency", "Frequency", "Monetary"] if c in df.columns]

    if rfm_cols_present:
        st.subheader("Distributions de Recency / Frequency / Monetary")
        d1, d2, d3 = st.columns(3)
        rfm_colors = {"Recency": "steelblue", "Frequency": "darkorange", "Monetary": "seagreen"}
        for col_widget, col_name in zip([d1, d2, d3], ["Recency", "Frequency", "Monetary"]):
            if col_name in df.columns:
                fig = px.histogram(
                    df, x=col_name, nbins=30, marginal="box",
                    title=f"Distribution de {col_name}",
                    color_discrete_sequence=[rfm_colors[col_name]],
                )
                col_widget.plotly_chart(fig, use_container_width=True)

    if "Segment" in df.columns:
        st.subheader("Segments RFM simples (score R+F+M)")
        s1, s2 = st.columns([1, 1])
        with s1:
            seg_counts = df["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Nombre de clients"]
            fig_seg = px.bar(
                seg_counts.sort_values("Nombre de clients"), x="Nombre de clients", y="Segment",
                orientation="h", title="Nombre de clients par segment RFM",
                color="Segment", color_discrete_map=color_map_for(seg_counts["Segment"]),
            )
            st.plotly_chart(fig_seg, use_container_width=True)
        with s2:
            if "Main_Offer" in df.columns:
                top_offers_list = df["Main_Offer"].value_counts().nlargest(10).index
                sub = df[df["Main_Offer"].isin(top_offers_list)]
                cross_tab = pd.crosstab(sub["Main_Offer"], sub["Segment"])
                fig_cross_seg = px.imshow(
                    cross_tab, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd",
                    title="Segments RFM par offre principale (top 10 offres)",
                    labels=dict(x="Segment", y="Offre", color="Nb clients"),
                )
                st.plotly_chart(fig_cross_seg, use_container_width=True)
    else:
        st.info("Colonne 'Segment' absente du CSV — regenere l'export depuis le notebook mis a jour.")

    if all(c in df.columns for c in ["F_score", "R_score" if "R_score" in df.columns else "F_score", "Monetary"]) and "R_score" in df.columns:
        st.subheader("Montant moyen depense selon Recence x Frequence")
        pivot_rfm = df.pivot_table(
            index="F_score", columns="R_score", values="Monetary", aggfunc="mean"
        ).sort_index(ascending=False)
        fig_heat = px.imshow(
            pivot_rfm, text_auto=".0f", color_continuous_scale="YlGnBu", aspect="auto",
            title="Montant moyen depense selon Recence x Frequence",
            labels=dict(x="R_score (5 = tres recent)", y="F_score (5 = tres frequent)", color="Monetary moyen"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    if "Monetary" in df.columns:
        st.subheader("Courbe de Lorenz : concentration du revenu")
        rfm_sorted = df.sort_values("Monetary", ascending=False).reset_index(drop=True)
        rfm_sorted["cum_clients_pct"] = (rfm_sorted.index + 1) / len(rfm_sorted) * 100
        rfm_sorted["cum_revenue_pct"] = rfm_sorted["Monetary"].cumsum() / rfm_sorted["Monetary"].sum() * 100
        idx_20 = max(int(len(rfm_sorted) * 0.2) - 1, 0)
        part_revenu_top20 = rfm_sorted.loc[idx_20, "cum_revenue_pct"]

        fig_lorenz = go.Figure()
        fig_lorenz.add_trace(go.Scatter(
            x=rfm_sorted["cum_clients_pct"], y=rfm_sorted["cum_revenue_pct"],
            mode="lines", name="Courbe de Lorenz", line=dict(color="crimson", width=3),
        ))
        fig_lorenz.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100], mode="lines", name="Repartition egalitaire",
            line=dict(dash="dash", color="grey"),
        ))
        fig_lorenz.add_trace(go.Scatter(
            x=[20], y=[part_revenu_top20], mode="markers+text", name="Top 20% clients",
            marker=dict(size=12, color="black"),
            text=[f"Top 20% clients = {part_revenu_top20:.0f}% du CA"],
            textposition="top right",
        ))
        fig_lorenz.update_layout(
            title="Concentration du chiffre d'affaires sur les meilleurs clients",
            xaxis_title="% cumule de clients (tries par Monetary decroissant)",
            yaxis_title="% cumule du chiffre d'affaires",
        )
        st.plotly_chart(fig_lorenz, use_container_width=True)
        st.caption(
            f"Les 20% de clients les plus depensiers representent a eux seuls "
            f"{part_revenu_top20:.0f}% du chiffre d'affaires total : la base clients "
            f"est fortement concentree en valeur."
        )

    if "Segment" in df.columns and "Monetary" in df.columns:
        st.subheader("Top clients par valeur (Champions)")
        champions_cols = [c for c in [
            "msisdn_crypte", "Main_Offer", "Recency", "Frequency", "Monetary", "RFM_Total"
        ] if c in df.columns]
        top_champions = df[df["Segment"] == "Champion"].sort_values("Monetary", ascending=False).head(10)
        if not top_champions.empty:
            st.dataframe(display_customer_table(top_champions[champions_cols]), use_container_width=True)
        else:
            st.caption("Aucun client dans le segment 'Champion' avec les filtres actuels.")

# ------------------------------------------------------------------------------
# ONGLET 4 : SEGMENTATION COMPORTEMENTALE AVANCEE
# ------------------------------------------------------------------------------
with tab_behavior:
    st.header("Segmentation comportementale avancee (RFM etendu)")

    if "Segment_Avance" in df.columns:
        profile_agg = {"msisdn_crypte": "count"}
        rename_map = {"msisdn_crypte": "Nb_clients"}
        for c, alias in [("Recency", "Recency_moy"), ("Frequency", "Frequency_moy"), ("Monetary", "Monetary_moy")]:
            if c in df.columns:
                profile_agg[c] = "mean"
                rename_map[c] = alias
        if "Monetary" in df.columns:
            profile_agg["Monetary"] = ["mean", "sum"]

        profil_segments = df.groupby("Segment_Avance").agg(
            Nb_clients=("msisdn_crypte", "count"),
            **({"Recency_moy": ("Recency", "mean")} if "Recency" in df.columns else {}),
            **({"Frequency_moy": ("Frequency", "mean")} if "Frequency" in df.columns else {}),
            **({"Monetary_moy": ("Monetary", "mean")} if "Monetary" in df.columns else {}),
            **({"Revenu_total": ("Monetary", "sum")} if "Monetary" in df.columns else {}),
        )

        if "Revenu_total" in profil_segments.columns:
            profil_segments = profil_segments.sort_values("Revenu_total", ascending=False)
            profil_segments["% Clients"] = (profil_segments["Nb_clients"] / profil_segments["Nb_clients"].sum() * 100).round(1)
            profil_segments["% Revenu"] = (profil_segments["Revenu_total"] / profil_segments["Revenu_total"].sum() * 100).round(1)

        st.subheader("Profil detaille de chaque segment")
        st.dataframe(profil_segments.round(2), use_container_width=True)

        if "% Clients" in profil_segments.columns and "% Revenu" in profil_segments.columns:
            st.subheader("Nombre de clients vs chiffre d'affaires")
            order = profil_segments.index.tolist()
            b1, b2 = st.columns(2)
            with b1:
                fig_pct_clients = px.bar(
                    profil_segments.loc[order].reset_index(), x="% Clients", y="Segment_Avance",
                    orientation="h", title="% de clients par segment",
                    color="Segment_Avance", color_discrete_map=color_map_for(order),
                )
                fig_pct_clients.update_layout(showlegend=False)
                st.plotly_chart(fig_pct_clients, use_container_width=True)
            with b2:
                fig_pct_rev = px.bar(
                    profil_segments.loc[order].reset_index(), x="% Revenu", y="Segment_Avance",
                    orientation="h", title="% du chiffre d'affaires par segment",
                    color="Segment_Avance", color_discrete_map=color_map_for(order),
                )
                fig_pct_rev.update_layout(showlegend=False)
                st.plotly_chart(fig_pct_rev, use_container_width=True)

        if all(c in df.columns for c in ["R_score", "F_score", "M_score"]):
            st.subheader("Profil R-F-M moyen par segment (radar)")
            radar_df = df.groupby("Segment_Avance")[["R_score", "F_score", "M_score"]].mean()
            categories = ["R_score", "F_score", "M_score"]

            fig_radar = go.Figure()
            for segment in radar_df.index:
                values = radar_df.loc[segment, categories].tolist()
                values += values[:1]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=categories + [categories[0]],
                    fill="toself", name=segment, opacity=0.6,
                    line=dict(color=color_map_for(radar_df.index)[segment]),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                title="Profil R-F-M moyen par segment comportemental",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        if rfm_cols_present := [c for c in ["Recency", "Frequency", "Monetary"] if c in df.columns]:
            st.subheader("Densite de Recency / Frequency / Monetary par segment")
            dens_cols = st.columns(len(rfm_cols_present))
            for col_widget, col_name in zip(dens_cols, rfm_cols_present):
                fig_dens = px.violin(
                    df, x="Segment_Avance", y=col_name, color="Segment_Avance",
                    color_discrete_map=color_map_for(df["Segment_Avance"].unique()),
                    box=True, title=f"Distribution de {col_name} par segment",
                )
                fig_dens.update_layout(showlegend=False, xaxis_tickangle=-30)
                col_widget.plotly_chart(fig_dens, use_container_width=True)

        if "Monetary" in df.columns and "Segment_Avance" in df.columns:
            st.subheader("Clients a risque (segment comportemental)")
            clients_a_risque = df[df["Segment_Avance"] == "Clients à risque"]
            r1, r2 = st.columns(2)
            r1.metric("Nombre de clients a risque", f"{len(clients_a_risque):,}")
            r2.metric("Chiffre d'affaires historique concerne", f"{clients_a_risque['Monetary'].sum():,.0f}")
            risk_cols = [c for c in [
                "msisdn_crypte", "Main_Offer", "Recency", "Frequency", "Monetary", "RFM_Total"
            ] if c in df.columns]
            if not clients_a_risque.empty:
                st.dataframe(
                    display_customer_table(
                        clients_a_risque.sort_values("Monetary", ascending=False).head(10)[risk_cols]
                    ),
                    use_container_width=True,
                )

        if "RFM_Weighted" in df.columns and "RFM_Total" in df.columns:
            st.subheader("Score simple (R+F+M) vs score pondere")
            df_rank = df.copy()
            df_rank["Rang_Total"] = df_rank["RFM_Total"].rank(ascending=False, method="min")
            df_rank["Rang_Weighted"] = df_rank["RFM_Weighted"].rank(ascending=False, method="min")
            fig_rank = px.scatter(
                df_rank, x="Rang_Total", y="Rang_Weighted", opacity=0.3,
                title="Score simple vs score pondere : qui change de classement ?",
                labels={"Rang_Total": "Rang (score simple R+F+M)", "Rang_Weighted": "Rang (score pondere)"},
            )
            lims = [1, len(df_rank)]
            fig_rank.add_trace(go.Scatter(
                x=lims, y=lims, mode="lines", name="Classements identiques",
                line=dict(dash="dash", color="red"),
            ))
            st.plotly_chart(fig_rank, use_container_width=True)
            st.caption(
                "Le score pondere (0.5 x montant + 0.3 x frequence + 0.2 x recence) donne plus "
                "de poids au montant depense : les points loin de la diagonale sont des clients "
                "dont le classement change significativement selon la methode utilisee."
            )
    else:
        st.info(
            "Colonne 'Segment_Avance' absente du CSV — regenere l'export depuis le notebook mis "
            "a jour pour voir cette section."
        )

# ------------------------------------------------------------------------------
# ONGLET 5 : CLUSTERING K-MEANS
# ------------------------------------------------------------------------------
with tab_cluster:
    st.header("Segments (K-Means)")

    if "Cluster_KMeans" in df.columns:
        c1, c2 = st.columns(2)

        with c1:
            size_df = df["Cluster_KMeans"].value_counts().sort_index().reset_index()
            size_df.columns = ["Cluster", "Nombre de clients"]
            fig_size = px.bar(size_df, x="Cluster", y="Nombre de clients",
                               title="Taille de chaque segment", color="Cluster")
            st.plotly_chart(fig_size, use_container_width=True)

        with c2:
            if "PC1" in df.columns and "PC2" in df.columns:
                fig_pca = px.scatter(df, x="PC1", y="PC2", color=df["Cluster_KMeans"].astype(str),
                                      title="Projection PCA coloree par cluster",
                                      labels={"color": "Cluster"})
                st.plotly_chart(fig_pca, use_container_width=True)
            else:
                st.info("Colonnes PC1/PC2 absentes du CSV.")

        profile_candidates = [c for c in [
            "Recency", "Frequency", "Monetary", "Tenure_Months",
            "Nb_Offers_Distinctes", "Life_Time", "Montant_Moyen_Par_Achat",
            "Frequency_Par_Mois", "Monetary_Par_Mois",
        ] if c in df.columns]

        if profile_candidates:
            profile = df.groupby("Cluster_KMeans")[profile_candidates].mean().round(1)
            st.subheader("Profil moyen par segment")
            st.dataframe(profile, use_container_width=True)

            st.subheader("Distribution de chaque variable par cluster")
            n_cols = 3
            rows = [profile_candidates[i:i + n_cols] for i in range(0, len(profile_candidates), n_cols)]
            for row_feats in rows:
                row_widgets = st.columns(len(row_feats))
                for w, feat in zip(row_widgets, row_feats):
                    fig_box = px.box(
                        df, x=df["Cluster_KMeans"].astype(str), y=feat,
                        color=df["Cluster_KMeans"].astype(str),
                        title=feat, labels={"x": "Cluster"},
                    )
                    fig_box.update_layout(showlegend=False)
                    w.plotly_chart(fig_box, use_container_width=True)

        if "Monetary" in df.columns:
            rev_by_cluster = df.groupby("Cluster_KMeans")["Monetary"].sum().reset_index()
            rev_by_cluster.columns = ["Cluster", "Revenu total"]
            fig_rev = px.pie(rev_by_cluster, names="Cluster", values="Revenu total",
                              title="Contribution au revenu par segment")
            st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.info("Colonne 'Cluster_KMeans' absente du CSV.")

# ------------------------------------------------------------------------------
# ONGLET 6 : RISQUE DE CHURN
# ------------------------------------------------------------------------------
with tab_churn:
    st.header("Risque de churn")

    if "Churn_Proba" in df.columns:

        def risk_bucket(p):
            if p >= 0.7:
                return "Eleve (>=0.70)"
            elif p >= 0.3:
                return "Moyen (0.30-0.69)"
            else:
                return "Faible (<0.30)"

        df["Risk_Bucket"] = df["Churn_Proba"].apply(risk_bucket)
        bucket_order = ["Faible (<0.30)", "Moyen (0.30-0.69)", "Eleve (>=0.70)"]
        bucket_colors = {
            "Faible (<0.30)": "#2ca02c",
            "Moyen (0.30-0.69)": "#ff9f1c",
            "Eleve (>=0.70)": "#d62728",
        }

        at_risk = df[df["Churn_Proba"] >= risk_threshold].sort_values("Churn_Proba", ascending=False)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Clients au-dessus du seuil", f"{len(at_risk):,}",delta_color="off")
        k2.metric("% de la base au-dessus du seuil", f"{len(at_risk) / len(df) * 100:.1f}%",delta_color="off")
        k3.metric("Churn proba moyenne", f"{df['Churn_Proba'].mean():.2f}",delta_color="off")
        k4.metric("Churn proba mediane", f"{df['Churn_Proba'].median():.2f}",delta_color="off")
        if "Monetary" in df.columns:
            revenue_at_risk = at_risk["Monetary"].sum()
            revenue_share = revenue_at_risk / df["Monetary"].sum() * 100 if df["Monetary"].sum() else 0
            k5.metric("Revenu expose (au-dessus du seuil)", f"{revenue_at_risk:,.0f}",
                       f"{revenue_share:.1f}% du revenu total",delta_color="off")
        else:
            k5.metric("Revenu expose", "N/A",delta_color="off")

        b1, b2, b3 = st.columns(3)
        bucket_counts = df["Risk_Bucket"].value_counts().reindex(bucket_order).fillna(0)
        b1.metric("Risque faible", f"{int(bucket_counts['Faible (<0.30)']):,}",
                   f"{bucket_counts['Faible (<0.30)'] / len(df) * 100:.1f}%",delta_color="off")
        b2.metric("Risque moyen", f"{int(bucket_counts['Moyen (0.30-0.69)']):,}",
                   f"{bucket_counts['Moyen (0.30-0.69)'] / len(df) * 100:.1f}%",delta_color="off")
        b3.metric("Risque eleve", f"{int(bucket_counts['Eleve (>=0.70)']):,}",
                   f"{bucket_counts['Eleve (>=0.70)'] / len(df) * 100:.1f}%",delta_color="off")

        st.markdown("")

        h1, h2 = st.columns([2, 1])

        with h1:
            fig_hist = px.histogram(
                df, x="Churn_Proba", nbins=30, color="Risk_Bucket",
                category_orders={"Risk_Bucket": bucket_order},
                color_discrete_map=bucket_colors,
                title="Distribution des probabilites de churn (par niveau de risque)",
                marginal="box",
            )
            fig_hist.add_vline(x=risk_threshold, line_dash="dash", line_color="black",
                                annotation_text="Seuil", annotation_position="top")
            st.plotly_chart(fig_hist, use_container_width=True)

        with h2:
            bucket_pie_df = bucket_counts.reset_index()
            bucket_pie_df.columns = ["Risque", "Nombre"]
            fig_bucket_pie = px.pie(
                bucket_pie_df,
                names="Risque", values="Nombre",
                category_orders={"Risque": bucket_order},
                color="Risque", color_discrete_map=bucket_colors,
                title="Repartition des niveaux de risque",
            )
            st.plotly_chart(fig_bucket_pie, use_container_width=True)

        if "Churned" in df.columns:
            st.subheader("Fiabilite du modele : calibration des probabilites")
            calib = df.copy()
            try:
                calib["Proba_Decile"] = pd.qcut(calib["Churn_Proba"], 10, duplicates="drop")
            except ValueError:
                calib["Proba_Decile"] = pd.cut(calib["Churn_Proba"], 10)

            calib_summary = (
                calib.groupby("Proba_Decile", observed=True)
                .agg(Proba_Moyenne_Predite=("Churn_Proba", "mean"),
                     Taux_Churn_Reel=("Churned", "mean"),
                     Nb_Clients=("Churned", "size"))
                .reset_index(drop=True)
            )

            fig_calib = go.Figure()
            fig_calib.add_trace(go.Scatter(
                x=calib_summary["Proba_Moyenne_Predite"], y=calib_summary["Taux_Churn_Reel"],
                mode="lines+markers", name="Modele",
                marker=dict(size=10, color="#1f77b4"),
                hovertemplate="Proba predite: %{x:.2f}<br>Churn reel: %{y:.2f}<extra></extra>",
            ))
            fig_calib.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite",
                line=dict(dash="dash", color="gray"),
            ))
            fig_calib.update_layout(
                title="Probabilite predite vs taux de churn reel (par decile)",
                xaxis_title="Probabilite de churn predite (moyenne du decile)",
                yaxis_title="Taux de churn reel observe",
                xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_calib, use_container_width=True)
            st.caption(
                "Plus la courbe bleue suit la diagonale, plus les probabilites predites "
                "reflètent fidelement le taux de churn reellement observe."
            )

        p1, p2 = st.columns(2)

        with p1:
            if "Cluster_KMeans" in df.columns:
                fig_box_cluster = px.box(
                    df, x=df["Cluster_KMeans"].astype(str), y="Churn_Proba", color=df["Cluster_KMeans"].astype(str),
                    title="Distribution du churn proba par segment",
                    labels={"x": "Cluster"},
                )
                st.plotly_chart(fig_box_cluster, use_container_width=True)

        with p2:
            if "Monetary" in df.columns:
                fig_scatter_risk = px.scatter(
                    df, x="Churn_Proba", y="Monetary",
                    color="Risk_Bucket", category_orders={"Risk_Bucket": bucket_order},
                    color_discrete_map=bucket_colors,
                    size="Frequency" if "Frequency" in df.columns else None,
                    hover_data=[c for c in ["msisdn_crypte", "Main_Offer"] if c in df.columns],
                    title="Exposition au risque : churn proba vs revenu client",
                )
                fig_scatter_risk.add_vline(x=risk_threshold, line_dash="dash", line_color="black")
                st.plotly_chart(fig_scatter_risk, use_container_width=True)

        display_cols = [c for c in [
            "msisdn_crypte", "Cluster_KMeans", "Segment_Avance", "Main_Offer", "Risk_Bucket",
            "Recency", "Frequency", "Monetary", "Churn_Proba",
        ] if c in df.columns]

        st.subheader(f"Top clients a risque (seuil >= {risk_threshold:.2f})")
        st.dataframe(
            display_customer_table(at_risk[display_cols].head(200)),
            use_container_width=True,
        )

        download_df = display_customer_table(at_risk[display_cols])
        st.download_button(
            "Telecharger la liste des clients a risque (CSV)",
            download_df.to_csv(index=False).encode("utf-8"),
            file_name="clients_a_risque.csv",
            mime="text/csv",
        )
    else:
        st.info("Colonne 'Churn_Proba' absente du CSV — regenere l'export depuis le notebook.")

# ------------------------------------------------------------------------------
# ONGLET 7 : MODELE PREDICTIF (comparaison + importance des variables)
# ------------------------------------------------------------------------------
with tab_model:
    st.header("Modele predictif de churn")

    if demo_mode:
        st.caption(
            "Les valeurs affichees dans cette demonstration sont illustratives et synthetiques ; "
            "elles ne correspondent pas aux resultats confidentiels de l'internship."
        )

    st.markdown(
        "Le pipeline compare **Random Forest**, **XGBoost** et **LightGBM**. "
        "Dans l'analyse originale, les modeles sont optimises avec `GridSearchCV` "
        "et validation croisee en utilisant le F1-score comme critere de selection."
    )

    if model_comparison_df is not None:
        st.subheader("Comparaison des modeles")
        st.dataframe(model_comparison_df.round(3), use_container_width=True)

        metric_cols = [c for c in ["Accuracy", "F1-score (classe 1)", "AUC-ROC"] if c in model_comparison_df.columns]
        if metric_cols and "Modele" in model_comparison_df.columns:
            melted = model_comparison_df.melt(id_vars="Modele", value_vars=metric_cols,
                                               var_name="Metrique", value_name="Score")
            fig_models = px.bar(
                melted, x="Modele", y="Score", color="Metrique", barmode="group",
                title="Accuracy / F1 / AUC-ROC par modele", range_y=[0, 1],
            )
            st.plotly_chart(fig_models, use_container_width=True)

        time_cols = [c for c in model_comparison_df.columns if "Temps" in c]
        if time_cols and "Modele" in model_comparison_df.columns:
            tcol1, tcol2 = st.columns(2)
            if "Temps GridSearchCV (s)" in model_comparison_df.columns:
                fig_time_train = px.bar(
                    model_comparison_df, x="Modele", y="Temps GridSearchCV (s)",
                    title="Temps d'entrainement (GridSearchCV)",
                )
                tcol1.plotly_chart(fig_time_train, use_container_width=True)
            pred_time_col = next((c for c in time_cols if "prédiction" in c or "prediction" in c), None)
            if pred_time_col:
                fig_time_pred = px.bar(
                    model_comparison_df, x="Modele", y=pred_time_col,
                    title=f"Temps de prediction ({pred_time_col})",
                )
                tcol2.plotly_chart(fig_time_pred, use_container_width=True)

        if "F1-score (classe 1)" in model_comparison_df.columns and "Modele" in model_comparison_df.columns:
            best_row = model_comparison_df.loc[model_comparison_df["F1-score (classe 1)"].idxmax()]
            if demo_mode:
                st.info(
                    f"Exemple synthetique : meilleur F1 affiche = **{best_row['Modele']}** "
                    f"({best_row['F1-score (classe 1)']:.3f}). "
                    "Cette valeur sert uniquement a illustrer le dashboard."
                )
            else:
                st.success(
                    f"Modele retenu pour scorer les clients : **{best_row['Modele']}** "
                    f"(F1-score = {best_row['F1-score (classe 1)']:.3f})"
                )
    else:
        st.info(
            "Fichier 'model_comparison.csv' absent — depose-le via le panneau de gauche "
            "(genere par la derniere cellule du notebook mis a jour) pour voir la comparaison "
            "des modeles."
        )

    st.divider()

    if feature_importance_df is not None:
        fcol = "Feature" if "Feature" in feature_importance_df.columns else feature_importance_df.columns[0]
        vcol = "Importance" if "Importance" in feature_importance_df.columns else feature_importance_df.columns[1]
        modele_gagnant = (
            feature_importance_df["Modele"].iloc[0]
            if "Modele" in feature_importance_df.columns and not feature_importance_df.empty
            else "modele retenu"
        )
        st.subheader(f"Importance des variables (modele {modele_gagnant})")
        fig_importance = px.bar(
            feature_importance_df.sort_values(vcol), x=vcol, y=fcol, orientation="h",
            title=f"Feature Importance - {modele_gagnant}",
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        st.caption(
            "Plus une variable est haute dans ce classement, plus elle a pese dans les "
            "decisions du modele pour distinguer un client churne d'un client fidele."
        )
    else:
        st.info(
            "Fichier 'feature_importance.csv' absent — depose-le via le panneau de gauche pour "
            "voir l'importance des variables du modele."
        )

# ------------------------------------------------------------------------------
# ONGLET 8 : OFFRES & REVENUS
# ------------------------------------------------------------------------------
with tab_offers:
    if "Main_Offer" in df.columns and "Monetary" in df.columns:
        st.header("Revenu par offre")
        rev_by_offer = (
            df.groupby("Main_Offer")["Monetary"].sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig_offer = px.bar(rev_by_offer, x="Main_Offer", y="Monetary",
                            title="Revenu total par offre principale (top 15)")
        st.plotly_chart(fig_offer, use_container_width=True)

        st.divider()
        st.header("KPIs detailles par offre")

        agg_dict = {}
        agg_labels = {}
        if "msisdn_crypte" in df.columns:
            agg_dict["msisdn_crypte"] = "count"
            agg_labels["msisdn_crypte"] = "Nb clients"
        if "Monetary" in df.columns:
            agg_dict["Monetary"] = "mean"
            agg_labels["Monetary"] = "ARPU (revenu moyen)"
        if "Churned" in df.columns:
            agg_dict["Churned"] = "mean"
            agg_labels["Churned"] = "Taux de churn"
        if "Churn_Proba" in df.columns:
            agg_dict["Churn_Proba"] = "mean"
            agg_labels["Churn_Proba"] = "Churn proba moyenne"
        if "Tenure_Months" in df.columns:
            agg_dict["Tenure_Months"] = "mean"
            agg_labels["Tenure_Months"] = "Anciennete moyenne (mois)"
        if "Frequency" in df.columns:
            agg_dict["Frequency"] = "mean"
            agg_labels["Frequency"] = "Frequence moyenne"
        if "Nb_Offers_Distinctes" in df.columns:
            agg_dict["Nb_Offers_Distinctes"] = "mean"
            agg_labels["Nb_Offers_Distinctes"] = "Nb offres distinctes moyen"

        offer_kpis = df.groupby("Main_Offer").agg(agg_dict).rename(columns=agg_labels)
        if "Taux de churn" in offer_kpis.columns:
            offer_kpis["Taux de churn"] = (offer_kpis["Taux de churn"] * 100).round(1)
        if "Churn proba moyenne" in offer_kpis.columns:
            offer_kpis["Churn proba moyenne"] = offer_kpis["Churn proba moyenne"].round(2)
        offer_kpis = offer_kpis.round(1).sort_values("Nb clients", ascending=False)

        st.dataframe(offer_kpis, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if "Taux de churn" in offer_kpis.columns:
                fig_churn_offer = px.bar(
                    offer_kpis.reset_index(), x="Main_Offer", y="Taux de churn",
                    title="Taux de churn par offre (%)"
                )
                st.plotly_chart(fig_churn_offer, use_container_width=True)
        with c2:
            if "ARPU (revenu moyen)" in offer_kpis.columns:
                fig_arpu = px.bar(
                    offer_kpis.reset_index(), x="Main_Offer", y="ARPU (revenu moyen)",
                    title="ARPU par offre"
                )
                st.plotly_chart(fig_arpu, use_container_width=True)

        if "Cluster_KMeans" in df.columns:
            st.subheader("Repartition des segments par offre (%)")
            cross = pd.crosstab(df["Main_Offer"], df["Cluster_KMeans"], normalize="index") * 100
            cross = cross.round(1)
            fig_cross = px.imshow(
                cross, text_auto=True, aspect="auto",
                labels=dict(x="Cluster", y="Offre", color="% clients"),
                title="Quel segment domine dans chaque offre"
            )
            st.plotly_chart(fig_cross, use_container_width=True)
    else:
        st.info("Colonnes 'Main_Offer' et/ou 'Monetary' absentes du CSV.")
